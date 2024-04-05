# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulate a quantum script."""
import copy

# pylint: disable=protected-access
from collections import Counter
from functools import singledispatch
from typing import Optional, Sequence

import numpy as np
from numpy.random import default_rng

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.transforms.dynamic_one_shot import gather_mcm
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples

INTERFACE_TO_LIKE = {
    # map interfaces known by autoray to themselves
    None: None,
    "numpy": "numpy",
    "autograd": "autograd",
    "jax": "jax",
    "torch": "torch",
    "tensorflow": "tensorflow",
    # map non-standard interfaces to those known by autoray
    "auto": None,
    "scipy": "numpy",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "pytorch": "torch",
    "tf": "tensorflow",
    "tensorflow-autograph": "tensorflow",
    "tf-autograph": "tensorflow",
}


class _FlexShots(qml.measurements.Shots):
    """Shots class that allows zero shots."""

    # pylint: disable=super-init-not-called
    def __init__(self, shots=None):
        if isinstance(shots, int):
            self.total_shots = shots
            self.shot_vector = (qml.measurements.ShotCopies(shots, 1),)
        else:
            self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])

        self._frozen = True


def _postselection_postprocess(state, is_state_batched, shots):
    """Update state after projector is applied."""
    if is_state_batched:
        raise ValueError(
            "Cannot postselect on circuits with broadcasting. Use the "
            "qml.transforms.broadcast_expand transform to split a broadcasted "
            "tape into multiple non-broadcasted tapes before executing if "
            "postselection is used."
        )

    # The floor function is being used here so that a norm very close to zero becomes exactly
    # equal to zero so that the state can become invalid. This way, execution can continue, and
    # bad postselection gives results that are invalid rather than results that look valid but
    # are incorrect.
    norm = qml.math.norm(state)

    if not qml.math.is_abstract(state) and qml.math.allclose(norm, 0.0):
        norm = 0.0

    if shots:
        # Clip the number of shots using a binomial distribution using the probability of
        # measuring the postselected state.
        postselected_shots = (
            [np.random.binomial(s, float(norm**2)) for s in shots]
            if not qml.math.is_abstract(norm)
            else shots
        )

        # _FlexShots is used here since the binomial distribution could result in zero
        # valid samples
        shots = _FlexShots(postselected_shots)

    state = state / norm
    return state, shots


def get_final_state(
    circuit, debugger=None, interface=None, initial_state=None, mid_measurements=None
):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        initial_state (TensorLike): Initial statevector
        mid_measurements (None, dict): Dictionary of mid-circuit measurements

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    if initial_state is None:
        circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    if initial_state is None:
        state = create_initial_state(
            sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface]
        )
    else:
        state = initial_state

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    for op in circuit.operations[bool(prep) :]:
        state = apply_operation(
            op,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            mid_measurements=mid_measurements,
        )
        # Handle postselection on mid-circuit measurements
        if isinstance(op, qml.Projector):
            state, circuit._shots = _postselection_postprocess(
                state, is_state_batched, circuit.shots
            )

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or (op.batch_size is not None)

    if initial_state is None:
        for _ in range(len(circuit.wires) - len(circuit.op_wires)):
            # if any measured wires are not operated on, we pad the state with zeros.
            # We know they belong at the end because the circuit is in standard wire-order
            state = qml.math.stack([state, qml.math.zeros_like(state)], axis=-1)

    return state, is_state_batched


# pylint: disable=too-many-arguments
def measure_final_state(
    circuit, state, is_state_batched, rng=None, prng_key=None, initial_state=None
) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.
        initial_state (TensorLike): Initial statevector

    Returns:
        Tuple[TensorLike]: The measurement results
    """
    if initial_state is None:
        circuit = circuit.map_to_standard_wires()

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    rng = default_rng(rng)
    results = measure_with_samples(
        circuit.measurements,
        state,
        shots=circuit.shots,
        is_state_batched=is_state_batched,
        rng=rng,
        prng_key=prng_key,
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)

        return results[0]

    return results


# pylint: disable=too-many-arguments
def simulate(
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
    state_cache: Optional[dict] = None,
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        state_cache=None (Optional[dict]): A dictionary mapping the hash of a circuit to the pre-rotated state. Used to pass the state between forward passes and vjp calculations.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    if circuit.shots and has_mid_circuit_measurements(circuit):
        return simulate_tree_mcm(
            circuit, rng=rng, prng_key=prng_key, debugger=debugger, interface=interface
        )
        # return simulate_one_shot_native_mcm(circuit, rng, prng_key, debugger, interface)
    state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface)
    if state_cache is not None:
        state_cache[circuit.hash] = state
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key)


# pylint: disable=too-many-arguments, dangerous-default-value
def simulate_tree_mcm(
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
    initial_state=None,
    mcm_active=None,
    mcm_samples=None,
) -> Result:
    """Simulate a single quantum script with native mid-circuit measurements.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        initial_state (TensorLike): Initial statevector
        mcm_active (dict): Mid-circuit measurement values or all parent circuits of ``circuit``
        mcm_samples (dict): Mid-circuit measurement samples or all parent circuits of ``circuit``

    Returns:
        tuple(TensorLike): The results of the simulation
    """

    #########################
    # shot vector treatment #
    #########################
    if circuit.shots.has_partitioned_shots:
        results = []
        for s in circuit.shots:
            aux_circuit = circuit.copy()
            aux_circuit._shots = qml.measurements.Shots(s)
            results.append(
                simulate_tree_mcm(
                    aux_circuit,
                    rng,
                    prng_key,
                    debugger,
                    interface,
                )
            )
        return tuple(results)

    #######################
    # main implementation #
    #######################

    def init_dict(d):
        return {} if d is None else d

    mcm_active = init_dict(mcm_active)
    mcm_samples = init_dict(mcm_samples)

    circuit_base, circuit_next, op = circuit_up_to_first_mcm(circuit)
    # we need to make sure the state is the all-wire state
    initial_state = prep_initial_state(circuit_base, interface, initial_state)
    state, is_state_batched = get_final_state(
        circuit_base,
        debugger=debugger,
        interface=interface,
        initial_state=initial_state,
        mid_measurements=mcm_active,
    )
    measurements = measure_final_state(
        circuit_base,
        state,
        is_state_batched,
        rng=rng,
        prng_key=prng_key,
        initial_state=initial_state,
    )

    # Simply return measurements when ``circuit_base`` does not have an MCM
    if circuit_next is None:
        return measurements

    # For 1-shot measurements as 1-D arrays
    samples = measurements.reshape((-1)) if measurements.ndim == 0 else measurements
    update_mcm_samples(op, samples, mcm_active, mcm_samples)

    # Define ``branch_measurement`` here to capture ``op``, ``rng``, ``prng_key``, ``debugger``, ``interface``
    def branch_measurement(
        circuit_base, circuit_next, counts, state, branch, mcm_active, mcm_samples
    ):
        """Returns the measurements of the specified branch by executing ``circuit_next``."""

        def branch_state(state, wire, branch):
            axis = wire.toarray()[0]
            slices = [slice(None)] * state.ndim
            slices[axis] = int(not branch)
            state = copy.deepcopy(state)
            state[tuple(slices)] = 0.0
            state_norm = np.linalg.norm(state)
            # we can throw here because vanished states should
            # be handled right outside ``branch_measurement``
            if state_norm < 1.0e-15:  # pragma: no cover
                raise ValueError(f"Cannot normalize state with state_norm {state_norm}")
            state = state / state_norm
            if op.reset and branch == 1:
                state = apply_operation(qml.PauliX(wire), state)
            return state

        wire = circuit_base._measurements[0].wires
        new_state = branch_state(state, wire, branch)
        circuit_next._shots = qml.measurements.Shots(counts[branch])
        return simulate_tree_mcm(
            circuit_next,
            rng=rng,
            prng_key=prng_key,
            debugger=debugger,
            interface=interface,
            initial_state=new_state,
            mcm_active=mcm_active,
            mcm_samples=mcm_samples,
        )

    counts = samples_to_counts(samples)
    measurements = []
    for branch in counts.keys():
        if op.postselect is not None and branch != op.postselect:
            prune_mcm_samples(op, branch, mcm_active, mcm_samples)
            continue
        mcm_active[op] = branch
        measurements.append(
            branch_measurement(
                circuit_base,
                circuit_next,
                counts,
                state,
                branch,
                mcm_active=mcm_active,
                mcm_samples=mcm_samples,
            )
        )
    measurements = dict(
        (
            (branch, (count, value))
            for branch, count, value in zip(counts.keys(), counts.values(), measurements)
        )
    )
    return combine_measurements(circuit, measurements, mcm_samples)


def samples_to_counts(samples):
    """Converts samples to counts.

    This function forces integer keys and values which are required by ``simulate_tree_mcm``.
    """
    counts = qml.math.unique(samples, return_counts=True)
    return dict((int(x), int(y)) for x, y in zip(*counts))


def prep_initial_state(circuit_base, interface, initial_state):
    """Returns an initial state which will act on all wires.

    ``get_final_state`` executes a circuit on a subset of wires found in operations
    or measurements, unless an initial_state is passed as an optional argument.
    This function makes sure that an initial state with the correct size is passed
    on the first invocation of ``simulate_tree_mcm``."""
    if initial_state is not None:
        return initial_state
    prep = None
    if len(circuit_base) > 0 and isinstance(circuit_base[0], qml.operation.StatePrepBase):
        prep = circuit_base[0]
    return create_initial_state(sorted(circuit_base.wires), prep, like=INTERFACE_TO_LIKE[interface])


def prune_mcm_samples(op, branch, mcm_active, mcm_samples):
    """Removes samples from mid-measurement sample dictionary given a MidMeasureMP and branch.

    Post-selection on a given mid-circuit measurement leads to ignoring certain branches
    of the tree and samples. The corresponding samples in all other mid-circuit measurement
    must be deleted accordingly. We need to find which samples are
    corresponding to the current branch by looking at all parent nodes.
    """
    mask = mcm_samples[op] == branch
    for k, v in mcm_active.items():
        if k == op:
            break
        mask = np.logical_and(mask, mcm_samples[k] == v)
    for k in mcm_samples.keys():
        mcm_samples[k] = mcm_samples[k][np.logical_not(mask)]


def update_mcm_samples(op, samples, mcm_active, mcm_samples):
    """Updates the mid-measurement sample dictionary given a MidMeasureMP and samples.

    If the ``mcm_active`` dictionary is empty, we are at the root and ``mcm_samples`
    is simply updated with ``samples``.

    If the ``mcm_active`` dictionary is not empty, we need to find which samples are
    corresponding to the current branch by looking at all parent nodes. ``mcm_samples`
    is then updated with samples at indices corresponding to parent nodes.
    """
    if mcm_active:
        shape = next(iter(mcm_samples.values())).shape
        mask = np.ones(shape, dtype=bool)
        for k, v in mcm_active.items():
            if k == op:
                break
            mask = np.logical_and(mask, mcm_samples[k] == v)
        if op not in mcm_samples:
            mcm_samples[op] = np.empty(shape, dtype=samples.dtype)
        mcm_samples[op][mask] = samples
    else:
        mcm_samples[op] = samples


def circuit_up_to_first_mcm(circuit):
    """Returns two circuits; one that runs up-to the next mid-circuit measurement and one that runs beyond it."""
    if not has_mid_circuit_measurements(circuit):
        return circuit, None, None

    # find next MidMeasureMP
    def find_next_mcm(circuit):
        for i, op in enumerate(circuit.operations):
            if isinstance(op, MidMeasureMP):
                return i, op
        return len(circuit.operations) + 1, None

    i, op = find_next_mcm(circuit)
    # run circuit until next MidMeasureMP and sample
    circuit_base = qml.tape.QuantumScript(
        circuit.operations,
        [qml.sample(wires=op.wires) if op.obs is None else qml.sample(op=op.obs)],
        shots=circuit.shots,
        trainable_params=circuit.trainable_params,
    )
    circuit_base._ops = circuit_base._ops[0:i]
    # circuit beyond next MidMeasureMP with VarianceMP <==> SampleMP
    new_measurements = []
    for m in circuit.measurements:
        if not m.mv:
            if isinstance(m, VarianceMP):
                new_measurements.append(SampleMP(obs=m.obs))
            else:
                new_measurements.append(m)
    circuit_next = qml.tape.QuantumScript(
        circuit.operations,
        new_measurements,
        shots=circuit.shots,
        trainable_params=circuit.trainable_params,
    )
    circuit_next._ops = circuit_next._ops[i + 1 :]

    return circuit_base, circuit_next, op


def measurement_with_no_shots(measurement):
    """Returns a NaN scalar or array of the correct size when executing an all-invalid-shot circuit."""
    return (
        np.nan * np.ones_like(measurement.eigvals())
        if isinstance(measurement, ProbabilityMP)
        else np.nan
    )


def combine_measurements(circuit, measurements, mcm_samples):
    """Returns combined measurement values of various types."""

    keys = list(measurements.keys())
    # convert dict-of-lists to list-of-dicts
    if keys and isinstance(measurements[keys[0]][1], Sequence):
        ds = [
            [(measurements[keys[i]][0], m) for m in measurements[keys[i]][1]]
            for i in range(len(measurements))
        ]
        new_measurements = [{keys[0]: m0, keys[1]: m1} for m0, m1 in zip(*ds)]
    else:
        new_measurements = [measurements]
    empty_mcm_samples = len(next(iter(mcm_samples.values()))) == 0
    if empty_mcm_samples and any(len(m) != 0 for m in mcm_samples.values()):
        raise ValueError("mcm_samples have inconsistent shapes.")
    # loop over measurements
    final_measurements = []
    for circ_meas in circuit.measurements:
        if circ_meas.mv and empty_mcm_samples:
            comb_meas = measurement_with_no_shots(circ_meas)
        elif circ_meas.mv:
            comb_meas = gather_mcm(circ_meas, mcm_samples)
        elif not new_measurements or not new_measurements[0]:
            if len(new_measurements) > 0:
                _ = new_measurements.pop(0)
            comb_meas = measurement_with_no_shots(circ_meas)
        else:
            comb_meas = combine_measurements_core(circ_meas, new_measurements.pop(0))
        if isinstance(circ_meas, SampleMP):
            comb_meas = qml.math.squeeze(comb_meas)
        final_measurements.append(comb_meas)
    # special treatment of var
    for i, (c, m) in enumerate(zip(circuit.measurements, final_measurements)):
        if not c.mv and isinstance(circuit.measurements[i], VarianceMP):
            final_measurements[i] = qml.math.var(m)
    return final_measurements[0] if len(final_measurements) == 1 else tuple(final_measurements)


@singledispatch
def combine_measurements_core(original_measurement, measures):  # pylint: disable=unused-argument
    """Returns the combined measurement value of a given type."""
    raise TypeError(
        f"Native mid-circuit measurement mode does not support {type(original_measurement).__name__}"
    )


@combine_measurements_core.register
def _(original_measurement: CountsMP, measures):  # pylint: disable=unused-argument
    keys = list(measures.keys())
    new_counts = Counter()
    for k in keys:
        new_counts.update(measures[k][1])
    return dict(new_counts)


@combine_measurements_core.register
def _(original_measurement: ExpectationMP, measures):  # pylint: disable=unused-argument
    cum_value = 0
    total_counts = 0
    for v in measures.values():
        cum_value += v[0] * v[1]
        total_counts += v[0]
    return cum_value / total_counts


@combine_measurements_core.register
def _(original_measurement: ProbabilityMP, measures):  # pylint: disable=unused-argument
    cum_value = 0
    total_counts = 0
    for v in measures.values():
        cum_value += v[0] * v[1]
        total_counts += v[0]
    return cum_value / total_counts


@combine_measurements_core.register
def _(original_measurement: SampleMP, measures):  # pylint: disable=unused-argument
    new_sample = tuple(m[1] for m in measures.values())
    return np.squeeze(np.concatenate(new_sample))


@combine_measurements_core.register
def _(original_measurement: VarianceMP, measures):  # pylint: disable=unused-argument
    new_sample = tuple(m[1] for m in measures.values())
    return np.squeeze(np.concatenate(new_sample))


def simulate_one_shot_native_mcm(
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
) -> Result:
    """Simulate a single shot of a single quantum script with native mid-circuit measurements.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation
        dict: The mid-circuit measurement results of the simulation
    """
    mcm_dict = {}
    state, is_state_batched = get_final_state(
        circuit, debugger=debugger, interface=interface, mid_measurements=mcm_dict
    )
    if not np.allclose(np.linalg.norm(state), 1.0):
        return None, mcm_dict
    return (
        measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key),
        mcm_dict,
    )


def has_mid_circuit_measurements(
    circuit: qml.tape.QuantumScript,
):
    """Returns True if the circuit contains a MidMeasureMP object and False otherwise.

    Args:
        circuit (QuantumTape): A QuantumScript

    Returns:
        bool: Whether the circuit contains a MidMeasureMP object
    """
    return any(isinstance(op, MidMeasureMP) for op in circuit.operations)
