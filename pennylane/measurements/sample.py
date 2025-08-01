# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the qml.sample measurement.
"""
from collections.abc import Sequence

import numpy as np

from pennylane import math
from pennylane.exceptions import MeasurementShapeError, QuantumFunctionError
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .counts import CountsMP
from .measurements import SampleMeasurement
from .mid_measure import MeasurementValue
from .process_samples import process_raw_samples


class SampleMP(SampleMeasurement):
    """Measurement process that returns the samples of a given observable. If no observable is
    provided then basis state samples are returned directly from the device.

    Please refer to :func:`pennylane.sample` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    _shortname = "sample"

    def __init__(self, obs=None, wires=None, eigvals=None, id=None):

        if isinstance(obs, MeasurementValue):
            super().__init__(obs=obs)
            return

        if isinstance(obs, Sequence):
            if not all(
                isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in obs
            ) and not all(math.is_abstract(o) for o in obs):
                raise QuantumFunctionError(
                    "Only sequences of single MeasurementValues can be passed with the op "
                    "argument. MeasurementValues manipulated using arithmetic operators cannot be "
                    "used when collecting statistics for a sequence of mid-circuit measurements."
                )

            super().__init__(obs=obs)
            return

        if wires is not None:
            if obs is not None:
                raise ValueError(
                    "Cannot specify the wires to sample if an observable is provided. The wires "
                    "to sample will be determined directly from the observable."
                )
            wires = Wires(wires)

        super().__init__(obs=obs, wires=wires, eigvals=eigvals, id=id)

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ):
        if shots is None:
            raise ValueError("finite shots are required to use SampleMP")
        sample_eigvals = n_wires is None or has_eigvals
        dtype = float if sample_eigvals else int

        if n_wires == 0:
            dim = num_device_wires
        elif sample_eigvals:
            dim = 1
        else:
            dim = n_wires

        shape = []
        if shots != 1:
            shape.append(shots)
        if dim != 1:
            shape.append(dim)
        return tuple(shape), dtype

    @property
    def numeric_type(self):
        if self.obs is None:
            # Computational basis samples
            return int
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        if not shots:
            raise MeasurementShapeError(
                "Shots are required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        if self.obs:
            num_values_per_shot = 1  # one single eigenvalue
        elif self.mv is not None:
            num_values_per_shot = 1 if isinstance(self.mv, MeasurementValue) else len(self.mv)
        else:
            # one value per wire
            num_values_per_shot = len(self.wires) if len(self.wires) > 0 else num_device_wires

        shape = []
        if shots != 1:
            shape.append(shots)
        if num_values_per_shot != 1:
            shape.append(num_values_per_shot)
        return tuple(shape)

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: None | tuple[int, ...] = None,
        bin_size: None | int = None,
    ):
        return process_raw_samples(
            self, samples, wire_order, shot_range=shot_range, bin_size=bin_size
        )

    def process_counts(self, counts: dict, wire_order: Wires):
        samples = []
        mapped_counts = self._map_counts(counts, wire_order)
        for outcome, count in mapped_counts.items():
            outcome_sample = self._compute_outcome_sample(outcome)
            if len(self.wires) == 1 and self.eigvals() is None:
                # For sampling wires, if only one wire is sampled, flatten the list
                outcome_sample = outcome_sample[0]
            samples.extend([outcome_sample] * count)

        return np.array(samples)

    def _map_counts(self, counts_to_map, wire_order) -> dict:
        """
        Args:
            counts_to_map: Dictionary where key is binary representation of the outcome and value is its count
            wire_order: Order of wires to which counts_to_map should be ordered in

        Returns:
            Dictionary where counts_to_map has been reordered according to wire_order
        """
        with QueuingManager.stop_recording():
            helper_counts = CountsMP(wires=self.wires, all_outcomes=False)
        return helper_counts.process_counts(counts_to_map, wire_order)

    def _compute_outcome_sample(self, outcome) -> list:
        """
        Args:
            outcome (str): The binary string representation of the measurement outcome.

        Returns:
            list: A list of outcome samples for given binary string.
                If eigenvalues exist, the binary outcomes are mapped to their corresponding eigenvalues.
        """
        if self.eigvals() is not None:
            eigvals = self.eigvals()
            return eigvals[int(outcome, 2)]

        return [int(bit) for bit in outcome]


def sample(
    op: Operator | MeasurementValue | Sequence[MeasurementValue] | None = None,
    wires=None,
) -> SampleMP:
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning raw samples. If no observable is provided then basis state samples are returned
    directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Operator or MeasurementValue): a quantum observable object. To get samples
            for mid-circuit measurements, ``op`` should be a ``MeasurementValue``.
        wires (Sequence[int] or int or None): the wires we wish to sample from; ONLY set wires if
            op is ``None``.

    Returns:
        SampleMP: Measurement process instance

    Raises:
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    .. note::

        QNodes that return samples cannot, in general, be differentiated, since the derivative
        with respect to a sample --- a stochastic process --- is ill-defined. An alternative
        approach would be to use single-shot expectation values. For example, instead of this:

        .. code-block:: python

            from functools import partial
            dev = qml.device("default.qubit")

            @partial(qml.set_shots, shots=10)
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(angle):
                qml.RX(angle, wires=0)
                return qml.sample(qml.PauliX(0))

            angle = qml.numpy.array(0.1)
            res = qml.jacobian(circuit)(angle)

        Consider using :func:`~pennylane.expval` and a sequence of single shots, like this:

        .. code-block:: python

            from functools import partial
            dev = qml.device("default.qubit")

            @partial(qml.set_shots, shots=[(1, 10)])
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(angle):
                qml.RX(angle, wires=0)
                return qml.expval(qml.PauliX(0))

            def cost(angle):
                return qml.math.hstack(circuit(angle))

            angle = qml.numpy.array(0.1)
            res = qml.jacobian(cost)(angle)

    **Example**

    .. code-block:: python3

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=4)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    array([ 1.,  1.,  1., -1.])

    If no observable is provided, then the raw basis state samples obtained
    from device are returned (e.g., for a qubit device, samples from the
    computational device are returned). In this case, ``wires`` can be specified
    so that sample results only include measurement results of the qubits of interest.

    .. code-block:: python3

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=4)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

    Executing this QNode:

    >>> circuit(0.5)
    array([[0, 1],
           [0, 0],
           [1, 1],
           [0, 0]])

    """
    return SampleMP(obs=op, wires=None if wires is None else Wires(wires))
