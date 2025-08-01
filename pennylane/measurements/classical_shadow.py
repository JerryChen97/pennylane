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
This module contains the qml.classical_shadow measurement.
"""
import copy
from collections.abc import Iterable, Sequence
from string import ascii_letters

import numpy as np

from pennylane import math
from pennylane.exceptions import MeasurementShapeError
from pennylane.operation import Operator
from pennylane.ops import RZ, Hadamard, I, X, Y, Z
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires, WiresLike

from .measurements import MeasurementTransform


class ClassicalShadowMP(MeasurementTransform):
    """Represents a classical shadow measurement process occurring at the end of a
    quantum variational circuit.

    Please refer to :func:`pennylane.classical_shadow` for detailed documentation.


    Args:
        wires (.Wires): The wires the measurement process applies to.
        seed (Union[int, None]): The seed used to generate the random measurements
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    _shortname = "shadow"

    def __init__(
        self,
        wires: WiresLike | None = None,
        seed: int | None = None,
        id: str | None = None,
    ):
        self.seed = seed
        super().__init__(wires=wires, id=id)

    def _flatten(self):
        metadata = (("wires", self.wires), ("seed", self.seed))
        return (None, None), metadata

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            self.seed,
            tuple(self.wires.tolist()),
        )

        return hash(fingerprint)

    def process(self, tape, device):
        """
        Returns the measured bits and recipes in the classical shadow protocol.

        The protocol is described in detail in the `classical shadows paper <https://arxiv.org/abs/2002.08953>`_.
        This measurement process returns the randomized Pauli measurements (the ``recipes``)
        that are performed for each qubit and snapshot as an integer:

        - 0 for Pauli X,
        - 1 for Pauli Y, and
        - 2 for Pauli Z.

        It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
        is sampled, and 1 if the -1 eigenvalue is sampled.

        The device shots are used to specify the number of snapshots. If ``T`` is the number
        of shots and ``n`` is the number of qubits, then both the measured bits and the
        Pauli measurements have shape ``(T, n)``.

        This implementation is device-agnostic and works by executing single-shot
        quantum tapes containing randomized Pauli observables. Devices should override this
        if they can offer cleaner or faster implementations.

        .. seealso:: :func:`~pennylane.classical_shadow`

        Args:
            tape (QuantumTape): the quantum tape to be processed
            device (pennylane.devices.LegacyDevice): the device used to process the quantum tape

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        wires = self.wires
        n_snapshots = device.shots
        seed = self.seed

        original_shots = device.shots
        original_shot_vector = device._shot_vector  # pylint: disable=protected-access

        try:
            device.shots = 1
            # slow implementation but works for all devices
            n_qubits = len(wires)
            mapped_wires = np.array(device.map_wires(wires))

            # seed the random measurement generation so that recipes
            # are the same for different executions with the same seed
            rng = np.random.RandomState(seed)
            recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))
            obs_list = [X, Y, Z]

            outcomes = np.zeros((n_snapshots, n_qubits))

            for t in range(n_snapshots):
                # compute rotations for the Pauli measurements
                rotations = [
                    rot
                    for wire_idx, wire in enumerate(wires)
                    for rot in obs_list[recipes[t][wire_idx]].compute_diagonalizing_gates(
                        wires=wire
                    )
                ]

                device.reset()
                device.apply(tape.operations, rotations=tape.diagonalizing_gates + rotations)

                outcomes[t] = device.generate_samples()[0][mapped_wires]
        finally:
            device.shots = original_shots
            device._shot_vector = original_shot_vector  # pylint: disable=protected-access

        return math.cast(math.stack([outcomes, recipes]), dtype=np.int8)

    def process_state_with_shots(
        self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None
    ):
        """Process the given quantum state with the given number of shots

        Args:
            state (Sequence[complex]): quantum state vector given as a rank-N tensor, where
                each dimension has size 2 and N is the number of wires.
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used. The random measurement outcomes
                in the form of bits will be generated from this argument, while the random recipes will be
                created from the ``seed`` argument provided to ``.ClassicalShadowsMP``.

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used. ``T`` is the number of shots,
            and ``n`` is the number of qubits.
        """
        wire_map = {w: i for i, w in enumerate(wire_order)}
        mapped_wires = [wire_map[w] for w in self.wires]
        n_qubits = len(mapped_wires)
        num_dev_qubits = len(state.shape)

        # seed the random measurement generation so that recipes
        # are the same for different executions with the same seed
        recipe_rng = np.random.RandomState(self.seed)
        recipes = recipe_rng.randint(0, 3, size=(shots, n_qubits))

        bit_rng = np.random.default_rng(rng)

        obs_list = np.stack(
            [
                X.compute_matrix(),
                Y.compute_matrix(),
                Z.compute_matrix(),
            ]
        )

        # the diagonalizing matrices corresponding to the Pauli observables above
        diag_list = np.stack(
            [
                Hadamard.compute_matrix(),
                Hadamard.compute_matrix() @ RZ.compute_matrix(-np.pi / 2),
                I.compute_matrix(),
            ]
        )
        obs = obs_list[recipes]
        diagonalizers = diag_list[recipes]

        # There's a significant speedup if we use the following iterative
        # process to perform the randomized Pauli measurements:
        #   1. Randomly generate Pauli observables for all snapshots for
        #      a single qubit (e.g. the first qubit).
        #   2. Compute the expectation of each Pauli observable on the first
        #      qubit by tracing out all other qubits.
        #   3. Sample the first qubit based on each Pauli expectation.
        #   4. For all snapshots, determine the collapsed state of the remaining
        #      qubits based on the sample result.
        #   4. Repeat iteratively until no qubits are remaining.
        #
        # Observe that after the first iteration, the second qubit will become the
        # "first" qubit in the process. The advantage to this approach as opposed to
        # simulataneously computing the Pauli expectations for each qubit is that
        # the partial traces are computed over iteratively smaller subsystems, leading
        # to a significant speed-up.

        # transpose the state so that the measured wires appear first
        unmeasured_wires = [i for i in range(num_dev_qubits) if i not in mapped_wires]
        transposed_state = np.transpose(state, axes=mapped_wires + unmeasured_wires)

        outcomes = np.zeros((shots, n_qubits))
        stacked_state = np.repeat(transposed_state[np.newaxis, ...], shots, axis=0)

        for active_qubit in range(n_qubits):
            # stacked_state loses a dimension each loop

            # trace out every qubit except the first
            num_remaining_qubits = num_dev_qubits - active_qubit
            conj_state_first_qubit = ascii_letters[num_remaining_qubits]
            stacked_dim = ascii_letters[num_remaining_qubits + 1]

            state_str = f"{stacked_dim}{ascii_letters[:num_remaining_qubits]}"
            conj_state_str = (
                f"{stacked_dim}{conj_state_first_qubit}{ascii_letters[1:num_remaining_qubits]}"
            )
            target_str = f"{stacked_dim}a{conj_state_first_qubit}"

            first_qubit_state = np.einsum(
                f"{state_str},{conj_state_str}->{target_str}",
                stacked_state,
                np.conj(stacked_state),
            )

            # sample the observables on the first qubit
            probs = (np.einsum("abc,acb->a", first_qubit_state, obs[:, active_qubit]) + 1) / 2
            samples = bit_rng.random(size=probs.shape) > probs
            outcomes[:, active_qubit] = samples

            # collapse the state of the remaining qubits; the next qubit in line
            # becomes the first qubit for the next iteration
            rotated_state = np.einsum(
                "ab...,acb->ac...", stacked_state, diagonalizers[:, active_qubit]
            )
            stacked_state = rotated_state[np.arange(shots), samples.astype(np.int8)]

            # re-normalize the collapsed state
            sum_indices = tuple(range(1, num_remaining_qubits))
            state_squared = np.abs(stacked_state) ** 2
            norms = np.sqrt(np.sum(state_squared, sum_indices, keepdims=True))
            stacked_state /= norms

        return np.stack([outcomes, recipes]).astype(np.int8)

    def process_density_matrix_with_shots(
        self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None
    ):
        """Process the given quantum state (density matrix) with the given number of shots

        Args:
            state (Sequence[complex]): quantum density matrix given as a rank-N tensor, where
                each dim has size 2 and N is twice the number of wires.
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used. The random measurement outcomes
                in the form of bits will be generated from this argument, while the random recipes will be
                created from the ``seed`` argument provided to ``.ClassicalShadowsMP``.

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        wire_map = {w: i for i, w in enumerate(wire_order)}
        mapped_wires = [wire_map[w] for w in self.wires]
        n_qubits = len(mapped_wires)
        num_dev_qubits = len(state.shape) // 2

        # seed the random measurement generation so that recipes
        # are the same for different executions with the same seed
        recipe_rng = np.random.RandomState(self.seed)
        recipes = recipe_rng.randint(0, 3, size=(shots, n_qubits))

        bit_rng = np.random.default_rng(rng)

        obs_list = np.array(
            [
                [[0, 1], [1, 0]],  # X
                [[0, -1j], [1j, 0]],  # Y
                [[1, 0], [0, -1]],  # Z
            ]
        )

        # the diagonalizing matrices corresponding to the Pauli observables above
        diag_list = np.array(
            [
                [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
                [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 + 0.5j, -0.5 + 0.5j]],
                [[1, 0], [0, 1]],
            ]
        )
        obs = obs_list[recipes]
        diagonalizers = diag_list[recipes]

        # transpose the state so that the measured wires appear first
        unmeasured_wires = [i for i in range(num_dev_qubits) if i not in mapped_wires]
        transposed_state = np.transpose(
            state,
            axes=mapped_wires
            + unmeasured_wires
            + [w + num_dev_qubits for w in mapped_wires]
            + [w + num_dev_qubits for w in unmeasured_wires],
        )

        outcomes = np.zeros((shots, n_qubits))
        stacked_state = np.repeat(transposed_state[np.newaxis, ...], shots, axis=0)

        for active_qubit in range(n_qubits):
            # stacked_state loses a dimension each loop

            # trace out every qubit except the first
            num_remaining_qubits = num_dev_qubits - active_qubit
            conj_state_first_qubit = ascii_letters[num_remaining_qubits]
            stacked_dim = ascii_letters[num_remaining_qubits + 1]
            remaining_dim = ascii_letters[:num_remaining_qubits]
            traced_dim = remaining_dim[1:]

            state_str = f"{stacked_dim}{remaining_dim}"
            conj_state_str = f"{conj_state_first_qubit}{traced_dim}"
            target_str = f"{stacked_dim}a{conj_state_first_qubit}"

            first_qubit_state = np.einsum(
                f"{state_str}{conj_state_str}->{target_str}",
                stacked_state,
            )

            # sample the observables on the first qubit
            probs = (np.einsum("abc,acb->a", first_qubit_state, obs[:, active_qubit]) + 1) / 2
            samples = bit_rng.random(size=probs.shape) > probs
            outcomes[:, active_qubit] = samples

            # collapse the state of the remaining qubits; the next qubit in line
            # becomes the first qubit for the next iteration
            U = diagonalizers[:, active_qubit]
            UT = np.stack([math.conjugate(math.transpose(m)) for m in U])

            # index labeling:
            # (s, vL, a) (s, a, ..., b, ...) (s, b, vR) -> (s, vL, ..., vR, ...)
            v_dim_left = ascii_letters[num_remaining_qubits + 2]
            v_dim_right = ascii_letters[num_remaining_qubits + 3]
            a_dagger_dim = ascii_letters[num_remaining_qubits + 4]

            U_str = f"{stacked_dim}{v_dim_left}a"
            rho_str = f"{stacked_dim}a{traced_dim}{a_dagger_dim}..."
            UT_str = f"{stacked_dim}{a_dagger_dim}{v_dim_right}"
            new_str = f"{stacked_dim}{v_dim_left}{traced_dim}{v_dim_right}..."
            rotated_state = np.einsum(
                f"{U_str}, {rho_str}, {UT_str}->{new_str}", U, stacked_state, UT
            )
            sampled_index = samples.astype(np.int8)
            stacked_state = rotated_state[np.arange(shots), sampled_index]
            stacked_state = np.stack(
                [
                    np.take(stacked_state[i], sampled_index[i], axis=num_remaining_qubits - 1)
                    for i in range(shots)
                ]
            )

            # re-normalize the collapsed state
            norms = np.einsum(
                f"{stacked_dim}{traced_dim}{traced_dim}->{stacked_dim}", stacked_state
            )
            norms = norms.reshape(norms.shape + (1,) * (2 * num_remaining_qubits - 2))
            stacked_state /= norms

        return np.stack([outcomes, recipes]).astype(np.int8)

    @property
    def samples_computational_basis(self):
        return False

    @property
    def numeric_type(self):
        return int

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ) -> tuple:
        return (2, shots, n_wires), np.int8

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple[int, int, int]:
        # otherwise, the return type requires a device
        if shots is None:
            raise MeasurementShapeError(
                "Shots must be specified to obtain the shape of a classical "
                "shadow measurement process."
            )

        # the first entry of the tensor represents the measured bits,
        # and the second indicate the indices of the unitaries used
        return (2, shots, len(self.wires))

    def __copy__(self):
        return self.__class__(
            seed=self.seed,
            wires=self._wires,
        )


class ShadowExpvalMP(MeasurementTransform):
    """Measures the expectation value of an operator using the classical shadow measurement process.

    Please refer to :func:`~pennylane.shadow_expval` for detailed documentation.

    Args:
        H (Operator, Sequence[Operator]): Operator or list of Operators to compute the expectation value over.
        seed (Union[int, None]): The seed used to generate the random measurements
        k (int): Number of equal parts to split the shadow's measurements to compute the median of means.
            ``k=1`` corresponds to simply taking the mean over all measurements.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    _shortname = "shadowexpval"

    def _flatten(self):
        metadata = (
            ("seed", self.seed),
            ("k", self.k),
        )
        return (self.H,), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], **dict(metadata))

    def __init__(
        self,
        H: Operator | Sequence,
        seed: int | None = None,
        k: int = 1,
        id: str | None = None,
    ):
        self.seed = seed
        self.H = H
        self.k = k
        super().__init__(id=id)

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(
        cls,
        H: Operator | Sequence,
        seed: int | None = None,
        k: int = 1,
        **kwargs,
    ):
        if cls._obs_primitive is None:  # pragma: no cover
            return type.__call__(cls, H=H, seed=seed, k=k, **kwargs)  # pragma: no cover
        return cls._obs_primitive.bind(H, seed=seed, k=k, **kwargs)

    def process(self, tape, device):
        from pennylane.shadows import (  # pylint: disable=import-outside-toplevel # tach-ignore
            ClassicalShadow,
        )

        bits, recipes = classical_shadow(wires=self.wires, seed=self.seed).process(tape, device)
        shadow = ClassicalShadow(bits, recipes, wire_map=self.wires.tolist())
        return shadow.expval(self.H, self.k)

    def process_state_with_shots(
        self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None
    ):
        """Process the given quantum state with the given number of shots

        Args:
            state (Sequence[complex]): quantum state
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.

        Returns:
            float: The estimate of the expectation value.
        """
        bits, recipes = classical_shadow(wires=self.wires, seed=self.seed).process_state_with_shots(
            state, wire_order, shots, rng=rng
        )
        # tach-ignore
        from pennylane.shadows import ClassicalShadow  # pylint:disable=import-outside-toplevel

        shadow = ClassicalShadow(bits, recipes, wire_map=self.wires.tolist())
        return shadow.expval(self.H, self.k)

    def process_density_matrix_with_shots(
        self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None
    ):
        """Process the given quantum state with the given number of shots

        Args:
            state (Sequence[complex]): quantum state
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.

        Returns:
            float: The estimate of the expectation value.
        """
        bits, recipes = classical_shadow(
            wires=self.wires, seed=self.seed
        ).process_density_matrix_with_shots(state, wire_order, shots, rng=rng)
        # tach-ignore
        from pennylane.shadows import (  # tach-ignore pylint: disable=import-outside-toplevel
            ClassicalShadow,
        )

        shadow = ClassicalShadow(bits, recipes, wire_map=self.wires.tolist())
        return shadow.expval(self.H, self.k)

    @property
    def samples_computational_basis(self):
        return False

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return () if isinstance(self.H, Operator) else (len(self.H),)

    @property
    def wires(self):
        r"""The wires the measurement process acts on.

        This is the union of all the Wires objects of the measurement.
        """
        if isinstance(self.H, Iterable):
            return Wires.all_wires([h.wires for h in self.H])

        return self.H.wires

    def queue(self, context=QueuingManager):
        """Append the measurement process to an annotated queue, making sure
        the observable is not queued"""
        Hs = self.H if isinstance(self.H, Iterable) else [self.H]
        for H in Hs:
            context.remove(H)
        context.append(self)

        return self

    def __copy__(self):
        H_copy = (
            [copy.copy(H) for H in self.H] if isinstance(self.H, Iterable) else copy.copy(self.H)
        )
        return self.__class__(
            H=H_copy,
            k=self.k,
            seed=self.seed,
        )


def shadow_expval(H, k=1, seed=None) -> ShadowExpvalMP:
    r"""Compute expectation values using classical shadows in a differentiable manner.

    The canonical way of computing expectation values is to simply average the expectation values for each local snapshot, :math:`\langle O \rangle = \sum_t \text{tr}(\rho^{(t)}O) / T`.
    This corresponds to the case ``k=1``. In the original work, `2002.08953 <https://arxiv.org/abs/2002.08953>`_, it has been proposed to split the ``T`` measurements into ``k`` equal
    parts to compute the median of means. For the case of Pauli measurements and Pauli observables, there is no advantage expected from setting ``k>1``.

    Args:
        H (Union[Iterable[Operator], Operator]): Observable or
            iterable of observables to compute the expectation value over.
        k (int): Number of equal parts to split the shadow's measurements to compute the median of means. ``k=1`` (default) corresponds to simply taking the mean over all measurements.
        seed (Union[None, int]):  Seed used to randomly sample Pauli measurements during the
            classical shadows protocol. If None, a random seed will be generated. If a tape with
            a ``shadow_expval`` measurement is copied, the seed will also be copied.
            Different seeds are still generated for different constructed tapes.

    Returns:
        ShadowExpvalMP: Measurement process instance

    .. note::

        This measurement uses the measurement :func:`~.pennylane.classical_shadow` and the class :class:`~.pennylane.ClassicalShadow` for post-processing
        internally to compute expectation values. In order to compute correct gradients using PennyLane's automatic differentiation,
        you need to use this measurement.

    **Example**

    .. code-block:: python3

        from functools import partial
        H = qml.Hamiltonian([1., 1.], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.X(1)])

        dev = qml.device("default.qubit", wires=range(2))
        @partial(qml.set_shots, shots=10000)
        @qml.qnode(dev)
        def circuit(x, obs):
            qml.Hadamard(0)
            qml.CNOT((0,1))
            qml.RX(x, wires=0)
            return qml.shadow_expval(obs)

        x = np.array(0.5, requires_grad=True)

    We can compute the expectation value of H as well as its gradient in the usual way.

    >>> circuit(x, H)
    array(1.8774)
    >>> qml.grad(circuit)(x, H)
    -0.44999999999999984

    In ``shadow_expval``, we can pass a list of observables. Note that each qnode execution internally performs one quantum measurement, so be sure
    to include all observables that you want to estimate from a single measurement in the same execution.

    >>> Hs = [H, qml.X(0), qml.Y(0), qml.Z(0)]
    >>> circuit(x, Hs)
    array([ 1.881 , -0.0312, -0.0027, -0.0087])
    >>> qml.jacobian(circuit)(x, Hs)
    array([-0.4518,  0.0174, -0.0216, -0.0063])
    """
    seed = seed or np.random.randint(2**30)
    return ShadowExpvalMP(H=H, seed=seed, k=k)


def classical_shadow(wires: WiresLike, seed=None) -> ClassicalShadowMP:
    """
    The classical shadow measurement protocol.

    The protocol is described in detail in the paper `Predicting Many Properties of a Quantum System from Very Few Measurements <https://arxiv.org/abs/2002.08953>`_.
    This measurement process returns the randomized Pauli measurements (the ``recipes``)
    that are performed for each qubit and snapshot as an integer:

    - 0 for Pauli X,
    - 1 for Pauli Y, and
    - 2 for Pauli Z.

    It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
    is sampled, and 1 if the -1 eigenvalue is sampled.

    The device shots are used to specify the number of snapshots. If ``T`` is the number
    of shots and ``n`` is the number of qubits, then both the measured bits and the
    Pauli measurements have shape ``(T, n)``.

    Args:
        wires (Sequence[int]): the wires to perform Pauli measurements on
        seed (Union[None, int]):  Seed used to randomly sample Pauli measurements during the
            classical shadows protocol. If None, a random seed will be generated. If a tape with
            a ``classical_shadow`` measurement is copied, the seed will also be copied.
            Different seeds are still generated for different constructed tapes.

    Returns:
        ClassicalShadowMP: measurement process instance

    **Example**

    Consider the following QNode that prepares a Bell state and performs a classical
    shadow measurement:

    .. code-block:: python3

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=5)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.classical_shadow(wires=[0, 1])

    Executing this QNode produces the sampled bits and the Pauli measurements used:

    >>> bits, recipes = circuit()
    >>> bits
    tensor([[0, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 1]], dtype=uint8, requires_grad=True)
    >>> recipes
    tensor([[2, 2],
            [0, 2],
            [1, 0],
            [0, 2],
            [0, 2]], dtype=uint8, requires_grad=True)

    .. details::
        :title: Usage Details

        Consider again the QNode in the above example. Since the Pauli observables are
        randomly sampled, executing this QNode again would produce different bits and Pauli recipes:

        >>> bits, recipes = circuit()
        >>> bits
        tensor([[0, 1],
                [0, 1],
                [0, 0],
                [0, 1],
                [1, 1]], dtype=uint8, requires_grad=True)
        >>> recipes
        tensor([[1, 0],
                [2, 1],
                [2, 2],
                [1, 0],
                [0, 0]], dtype=uint8, requires_grad=True)

        To use the same Pauli recipes for different executions, the :class:`~.tape.QuantumTape`
        interface should be used instead:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            ops = [qml.Hadamard(wires=0), qml.CNOT(wires=(0,1))]
            measurements = [qml.classical_shadow(wires=(0,1))]
            tape = qml.tape.QuantumTape(ops, measurements, shots=5)

        >>> bits1, recipes1 = qml.execute([tape], device=dev, diff_method=None)[0]
        >>> bits2, recipes2 = qml.execute([tape], device=dev, diff_method=None)[0]
        >>> np.all(recipes1 == recipes2)
        True
        >>> np.all(bits1 == bits2)
        False

        If using different Pauli recipes is desired for the :class:`~.tape.QuantumTape` interface,
        different seeds should be used for the classical shadow:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            measurements1 = [qml.classical_shadow(wires=(0,1), seed=10)]
            tape1 = qml.tape.QuantumTape(ops, measurements1, shots=5)

            measurements2 = [qml.classical_shadow(wires=(0,1), seed=15)]
            tape2 = qml.tape.QuantumTape(ops, measurements2, shots=5)

        >>> bits1, recipes1 = qml.execute([tape1], device=dev, diff_method=None)[0]
        >>> bits2, recipes2 = qml.execute([tape2], device=dev, diff_method=None)[0]
        >>> np.all(recipes1 == recipes2)
        False
        >>> np.all(bits1 == bits2)
        False
    """
    wires = Wires(wires)
    seed = seed or np.random.randint(2**30)
    return ClassicalShadowMP(wires=wires, seed=seed)


if ShadowExpvalMP._obs_primitive is not None:  # pylint: disable=protected-access

    @ShadowExpvalMP._obs_primitive.def_impl  # pylint: disable=protected-access
    def _(H, **kwargs):
        return type.__call__(ShadowExpvalMP, H, **kwargs)
