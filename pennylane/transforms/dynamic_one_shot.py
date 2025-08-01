# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the batch dimension transform.
"""

import itertools
from collections import Counter
from collections.abc import Sequence

import numpy as np

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementValue,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn, TensorLike

from .core import transform

fill_in_value = np.iinfo(np.int32).min


def is_mcm(operation):
    """Returns True if the operation is a mid-circuit measurement and False otherwise."""
    mcm = isinstance(operation, MidMeasureMP)
    return mcm or "MidCircuitMeasure" in str(type(operation))


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def dynamic_one_shot(tape: QuantumScript, **kwargs) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transform a QNode to into several one-shot tapes to support dynamic circuit execution.

    This transform enables the ``"one-shot"`` mid-circuit measurement method. The ``"one-shot"`` method prompts the
    device to perform a series of one-shot executions, where in each execution, the ``qml.measure``
    operation applies a probabilistic mid-circuit measurement to the circuit.
    This is in contrast with ``qml.defer_measurement``, which instead introduces an extra
    wire for each mid-circuit measurement. The ``"one-shot"`` method is favourable in the few-shots
    and several-mid-circuit-measurements limit, whereas ``qml.defer_measurements`` is favourable in
    the opposite limit.

    Args:
        tape (QNode or QuantumScript or Callable): a quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
        This circuit will provide the results of a dynamic execution.

    **Example**

    Most devices that support mid-circuit measurements will include this transform in its
    preprocessing automatically when applicable. When this is the case, any user-applied
    ``dynamic_one_shot`` transforms will be ignored. The recommended way to use dynamic one
    shot is to specify ``mcm_method="one-shot"`` in the ``qml.qnode`` decorator.

    .. code-block:: python

        dev = qml.device("default.qubit")
        params = np.pi / 4 * np.ones(2)

        @partial(qml.set_shots, shots=100)
        @qml.qnode(dev, mcm_method="one-shot", postselect_mode="fill-shots")
        def func(x, y):
            qml.RX(x, wires=0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.RY)(y, wires=1)
            return qml.expval(op=m0)

    """
    if not any(is_mcm(o) for o in tape.operations):
        return (tape,), null_postprocessing

    for m in tape.measurements:
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} "
                "measurements."
            )
    _ = kwargs.get("device", None)

    if not tape.shots:
        raise QuantumFunctionError("dynamic_one_shot is only supported with finite shots.")

    samples_present = any(isinstance(mp, SampleMP) for mp in tape.measurements)
    postselect_present = any(op.postselect is not None for op in tape.operations if is_mcm(op))
    if postselect_present and samples_present and tape.batch_size is not None:
        raise ValueError(
            "Returning qml.sample is not supported when postselecting mid-circuit "
            "measurements with broadcasting"
        )

    if (batch_size := tape.batch_size) is not None:
        tapes, broadcast_fn = qml.transforms.broadcast_expand(tape)
    else:
        tapes = [tape]
        broadcast_fn = None

    aux_tapes = [init_auxiliary_tape(t) for t in tapes]

    postselect_mode = kwargs.get("postselect_mode", None)

    def reshape_data(array):
        return qml.math.squeeze(qml.math.vstack(array))

    def processing_fn(results, has_partitioned_shots=None, batched_results=None):
        if batched_results is None and batch_size is not None:
            # If broadcasting, recursively process the results for each batch. For each batch
            # there are tape.shots.total_shots results. The length of the first axis of final_results
            # will be batch_size.
            final_results = []
            for result in results:
                final_results.append(processing_fn((result,), batched_results=False))
            return broadcast_fn(final_results)

        if has_partitioned_shots is None and tape.shots.has_partitioned_shots:
            # If using shot vectors, recursively process the results for each shot bin. The length
            # of the first axis of final_results will be the length of the shot vector.
            results = list(results[0])
            final_results = []
            for s in tape.shots:
                final_results.append(
                    processing_fn(results[0:s], has_partitioned_shots=False, batched_results=False)
                )
                del results[0:s]
            return tuple(final_results)
        if not tape.shots.has_partitioned_shots:
            results = results[0]

        is_scalar = not isinstance(results[0], Sequence)
        if is_scalar:
            results = [reshape_data(tuple(results))]
        else:
            results = [
                reshape_data(tuple(res[i] for res in results)) for i, _ in enumerate(results[0])
            ]
        return parse_native_mid_circuit_measurements(
            tape, aux_tapes, results, postselect_mode=postselect_mode
        )

    return aux_tapes, processing_fn


def get_legacy_capabilities(dev):
    """Gets the capabilities dictionary of a device."""
    assert isinstance(dev, qml.devices.LegacyDeviceFacade)
    return dev.target_device.capabilities()


def _supports_one_shot(dev: "qml.devices.Device"):
    """Checks whether a device supports one-shot."""

    if isinstance(dev, qml.devices.LegacyDevice):
        return get_legacy_capabilities(dev).get("supports_mid_measure", False)

    return dev.name in ("default.qubit", "lightning.qubit") or (
        dev.capabilities is not None and "one-shot" in dev.capabilities.supported_mcm_methods
    )


@dynamic_one_shot.custom_qnode_transform
def _dynamic_one_shot_qnode(self, qnode, targs, tkwargs):
    """Custom qnode transform for ``dynamic_one_shot``."""
    if tkwargs.get("device", None):
        raise ValueError(
            "Cannot provide a 'device' value directly to the dynamic_one_shot decorator "
            "when transforming a QNode."
        )
    if qnode.device is not None:
        if not _supports_one_shot(qnode.device):
            raise TypeError(
                f"Device {qnode.device.name} does not support mid-circuit measurements and/or "
                "one-shot execution mode natively, and hence it does not support the "
                "dynamic_one_shot transform. 'default.qubit' and 'lightning.qubit' currently "
                "support mid-circuit measurements and the dynamic_one_shot transform."
            )
    tkwargs.setdefault("device", qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)


def init_auxiliary_tape(circuit: qml.tape.QuantumScript):
    """Creates an auxiliary circuit to perform one-shot mid-circuit measurement calculations.

    Measurements are replaced by SampleMP measurements on wires and observables found in the
    original measurements.

    Args:
        circuit (QuantumTape): The original QuantumScript

    Returns:
        QuantumScript: A copy of the circuit with modified measurements
    """
    new_measurements = []
    for m in circuit.measurements:
        if m.mv is None:
            if isinstance(m, VarianceMP):
                new_measurements.append(SampleMP(obs=m.obs))
            else:
                new_measurements.append(m)
    for op in circuit.operations:
        if "MidCircuitMeasure" in str(type(op)):  # pragma: no cover
            new_measurements.append(qml.sample(op.out_classical_tracers[0]))
        elif isinstance(op, MidMeasureMP):
            new_measurements.append(qml.sample(MeasurementValue([op], lambda res: res)))
    return qml.tape.QuantumScript(
        circuit.operations,
        new_measurements,
        shots=[1] * circuit.shots.total_shots,
        trainable_params=circuit.trainable_params,
    )


# pylint: disable=too-many-branches
def parse_native_mid_circuit_measurements(
    circuit: qml.tape.QuantumScript,
    aux_tapes: qml.tape.QuantumScript,
    results: TensorLike,
    postselect_mode=None,
):
    """Combines, gathers and normalizes the results of native mid-circuit measurement runs.

    Args:
        circuit (QuantumTape): The original ``QuantumScript``.
        aux_tapes (List[QuantumTape]): List of auxiliary ``QuantumScript`` objects.
        results (TensorLike): Array of measurement results.

    Returns:
        tuple(TensorLike): The results of the simulation.
    """

    def measurement_with_no_shots(measurement):
        return (
            np.nan * np.ones_like(measurement.eigvals())
            if isinstance(measurement, ProbabilityMP)
            else np.nan
        )

    interface = qml.math.get_deep_interface(results)
    interface = "numpy" if interface == "builtins" else interface
    interface = "tensorflow" if interface == "tf" else interface
    active_qjit = qml.compiler.active()

    all_mcms = [op for op in aux_tapes[0].operations if is_mcm(op)]
    n_mcms = len(all_mcms)
    mcm_samples = qml.math.hstack(
        tuple(qml.math.reshape(res, (-1, 1)) for res in results[-n_mcms:])
    )
    mcm_samples = qml.math.array(mcm_samples, like=interface)
    # Can't use boolean dtype array with tf, hence why conditionally setting items to 0 or 1
    has_postselect = qml.math.array(
        [[op.postselect is not None for op in all_mcms]],
        like=interface,
        dtype=mcm_samples.dtype,
    )
    postselect = qml.math.array(
        [[0 if op.postselect is None else op.postselect for op in all_mcms]],
        like=interface,
        dtype=mcm_samples.dtype,
    )
    is_valid = qml.math.all(mcm_samples * has_postselect == postselect, axis=1)
    has_valid = qml.math.any(is_valid)
    mid_meas = [op for op in circuit.operations if is_mcm(op)]
    mcm_samples = [mcm_samples[:, i : i + 1] for i in range(n_mcms)]
    mcm_samples = dict(zip(mid_meas, mcm_samples))
    normalized_meas = []
    m_count = 0
    for m in circuit.measurements:
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} measurements."
            )
        if interface != "jax" and m.mv is not None and not has_valid:
            meas = measurement_with_no_shots(m)
        elif m.mv is not None and active_qjit:
            meas = gather_mcm_qjit(
                m, mcm_samples, is_valid, postselect_mode=postselect_mode
            )  # pragma: no cover
        elif m.mv is not None:
            meas = gather_mcm(m, mcm_samples, is_valid, postselect_mode=postselect_mode)
        elif interface != "jax" and not has_valid:
            meas = measurement_with_no_shots(m)
            m_count += 1
        else:
            result = results[m_count]
            if not isinstance(m, CountsMP):
                # We don't need to cast to arrays when using qml.counts. qml.math.array is not viable
                # as it assumes all elements of the input are of builtin python types and not belonging
                # to any particular interface
                result = qml.math.array(result, like=interface)
            if active_qjit:  # pragma: no cover
                # `result` contains (bases, counts) need to return (basis, sum(counts)) where `is_valid`
                # Any row of `result[0]` contains basis, so we return `result[0][0]`
                # We return the sum of counts (`result[1]`) weighting by `is_valid`, which is `0` for invalid samples
                if isinstance(m, CountsMP):
                    normalized_meas.append(
                        (
                            result[0][0],
                            qml.math.sum(result[1] * qml.math.reshape(is_valid, (-1, 1)), axis=0),
                        )
                    )
                    m_count += 1
                    continue
                result = qml.math.squeeze(result)
            meas = gather_non_mcm(m, result, is_valid, postselect_mode=postselect_mode)
            m_count += 1
        if isinstance(m, SampleMP):
            meas = qml.math.squeeze(meas)
        normalized_meas.append(meas)

    return tuple(normalized_meas) if len(normalized_meas) > 1 else normalized_meas[0]


def gather_mcm_qjit(measurement, samples, is_valid, postselect_mode=None):  # pragma: no cover
    """Process MCM measurements when the Catalyst compiler is active.

    Args:
        measurement (MeasurementProcess): measurement
        samples (dict): Mid-circuit measurement samples
        is_valid (TensorLike): Boolean array with the same shape as ``samples`` where the value at
            each index specifies whether or not the respective sample is valid.

    Returns:
        TensorLike: The combined measurement outcome
    """
    found, meas = False, None
    for k, meas in samples.items():
        if measurement.mv is k.out_classical_tracers[0]:
            found = True
            break
    if not found:
        raise LookupError("MCM not found")
    meas = qml.math.squeeze(meas)
    if isinstance(measurement, (CountsMP, ProbabilityMP)):
        interface = qml.math.get_interface(is_valid)
        sum_valid = qml.math.sum(is_valid)
        count_1 = qml.math.sum(meas * is_valid)
        if isinstance(measurement, CountsMP):
            return qml.math.array([0, 1], like=interface), qml.math.array(
                [sum_valid - count_1, count_1], like=interface
            )
        if isinstance(measurement, ProbabilityMP):
            counts = qml.math.array([sum_valid - count_1, count_1], like=interface)
            return counts / sum_valid
    return gather_non_mcm(measurement, meas, is_valid, postselect_mode=postselect_mode)


def gather_non_mcm(measurement, samples, is_valid, postselect_mode=None):
    """Combines, gathers and normalizes several measurements with trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (TensorLike): Post-processed measurement samples
        is_valid (TensorLike): Boolean array with the same shape as ``samples`` where the value at
            each index specifies whether or not the respective sample is valid.

    Returns:
        TensorLike: The combined measurement outcome
    """
    if isinstance(measurement, CountsMP):
        tmp = Counter()

        if measurement.all_outcomes:
            if isinstance(measurement.mv, Sequence):
                values = [list(m.branches.values()) for m in measurement.mv]
                values = list(itertools.product(*values))
                tmp = Counter({"".join(map(str, v)): 0 for v in values})
            else:
                values = [list(measurement.mv.branches.values())]
                values = list(itertools.product(*values))
                tmp = Counter({float(*v): 0 for v in values})

        for i, d in enumerate(samples):
            tmp.update(
                {k if isinstance(k, str) else float(k): v * is_valid[i] for k, v in d.items()}
            )

        if not measurement.all_outcomes:
            tmp = Counter({k: v for k, v in tmp.items() if v > 0})
        return dict(sorted(tmp.items()))

    if isinstance(measurement, SampleMP):
        if postselect_mode == "pad-invalid-samples" and samples.ndim == 2:
            is_valid = qml.math.reshape(is_valid, (-1, 1))
        if postselect_mode == "pad-invalid-samples":
            return qml.math.where(is_valid, samples, fill_in_value)
        if qml.math.shape(samples) == ():  # single shot case
            samples = qml.math.reshape(samples, (-1, 1))
        return samples[is_valid]

    if (interface := qml.math.get_interface(is_valid)) == "tensorflow":
        # Tensorflow requires arrays that are used for arithmetic with each other to have the
        # same dtype. We don't cast if measuring samples as float tf.Tensors cannot be used to
        # index other tf.Tensors (is_valid is used to index valid samples).
        is_valid = qml.math.cast_like(is_valid, samples)

    if isinstance(measurement, ExpectationMP):
        return qml.math.sum(samples * is_valid) / qml.math.sum(is_valid)
    if isinstance(measurement, ProbabilityMP):
        return qml.math.sum(samples * qml.math.reshape(is_valid, (-1, 1)), axis=0) / qml.math.sum(
            is_valid
        )

    # VarianceMP
    expval = qml.math.sum(samples * is_valid) / qml.math.sum(is_valid)
    if interface == "tensorflow":
        # Casting needed for tensorflow
        samples = qml.math.cast_like(samples, expval)
        is_valid = qml.math.cast_like(is_valid, expval)
    return qml.math.sum((samples - expval) ** 2 * is_valid) / qml.math.sum(is_valid)


def gather_mcm(measurement, samples, is_valid, postselect_mode=None):
    """Combines, gathers and normalizes several measurements with non-trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (List[dict]): Mid-circuit measurement samples
        is_valid (TensorLike): Boolean array with the same shape as ``samples`` where the value at
            each index specifies whether or not the respective sample is valid.

    Returns:
        TensorLike: The combined measurement outcome
    """
    interface = qml.math.get_deep_interface(is_valid)
    mv = measurement.mv
    # The following block handles measurement value lists, like ``qml.counts(op=[mcm0, mcm1, mcm2])``.
    if isinstance(measurement, (CountsMP, ProbabilityMP, SampleMP)) and isinstance(mv, Sequence):
        mcm_samples = [m.concretize(samples) for m in mv]
        mcm_samples = qml.math.concatenate(mcm_samples, axis=1)
        if isinstance(measurement, ProbabilityMP):
            values = [list(m.branches.values()) for m in mv]
            values = list(itertools.product(*values))
            values = [qml.math.array([v], like=interface, dtype=mcm_samples.dtype) for v in values]
            # Need to use boolean functions explicitly as Tensorflow does not allow integer math
            # on boolean arrays
            counts = [
                qml.math.count_nonzero(
                    qml.math.logical_and(qml.math.all(mcm_samples == v, axis=1), is_valid)
                )
                for v in values
            ]
            counts = qml.math.array(counts, like=interface)
            return counts / qml.math.sum(counts)
        if isinstance(measurement, CountsMP):
            mcm_samples = [{"".join(str(int(v)) for v in tuple(s)): 1} for s in mcm_samples]
        return gather_non_mcm(measurement, mcm_samples, is_valid, postselect_mode=postselect_mode)
    mcm_samples = qml.math.ravel(qml.math.array(mv.concretize(samples), like=interface))
    if isinstance(measurement, ProbabilityMP):
        # Need to use boolean functions explicitly as Tensorflow does not allow integer math
        # on boolean arrays
        counts = [
            qml.math.count_nonzero(qml.math.logical_and((mcm_samples == v), is_valid))
            for v in list(mv.branches.values())
        ]
        counts = qml.math.array(counts, like=interface)
        return counts / qml.math.sum(counts)
    if isinstance(measurement, CountsMP):
        mcm_samples = [{float(s): 1} for s in mcm_samples]
    return gather_non_mcm(measurement, mcm_samples, is_valid, postselect_mode=postselect_mode)
