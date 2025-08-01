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
A transform for decomposing quantum circuits into user defined gate sets. Offers an alternative to the more device-focused decompose transform.
"""


from __future__ import annotations

import warnings
from collections import ChainMap
from collections.abc import Callable, Generator, Iterable, Sequence
from functools import lru_cache, partial

import pennylane as qml
from pennylane.decomposition import DecompositionGraph
from pennylane.decomposition.utils import translate_op_alias
from pennylane.operation import Operator
from pennylane.transforms.core import transform


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@lru_cache
def _get_plxpr_decompose():  # pylint: disable=missing-docstring, too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        import jax

        from pennylane.capture.primitives import ctrl_transform_prim
        from pennylane.decomposition.collect_resource_ops import CollectResourceOps

    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name, too-few-public-methods

    class ControlTransformInterpreter(qml.capture.PlxprInterpreter):
        """Interpreter for replacing control transforms with individually controlled ops."""

        def __init__(self, control_wires, control_values=None, work_wires=None):
            super().__init__()
            self.control_wires = control_wires
            self.control_values = control_values
            self.work_wires = work_wires

        def interpret_operation(self, op):
            """Interpret operation."""
            with qml.capture.pause():
                ctrl_op = qml.ctrl(
                    op,
                    self.control_wires,
                    control_values=self.control_values,
                    work_wires=self.work_wires,
                )
            super().interpret_operation(ctrl_op)

    # pylint: disable=too-many-instance-attributes
    class DecomposeInterpreter(qml.capture.PlxprInterpreter):
        """Plxpr Interpreter for applying the ``decompose`` transform to callables or jaxpr
        when program capture is enabled.
        """

        def __init__(
            self,
            *,
            gate_set=None,
            stopping_condition=None,
            max_expansion=None,
            fixed_decomps=None,
            alt_decomps=None,
        ):  # pylint: disable=too-many-arguments

            self.max_expansion = max_expansion
            self._current_depth = 0

            if not qml.decomposition.enabled_graph() and (fixed_decomps or alt_decomps):
                raise TypeError(
                    "The keyword arguments fixed_decomps and alt_decomps are only available with "
                    "the new experimental graph-based decomposition system. Use qml.decomposition.enable_graph() "
                    "to enable the new system."
                )

            self._decomp_graph = None
            self._target_gate_names = None
            self._fixed_decomps, self._alt_decomps = fixed_decomps, alt_decomps

            # We use a ChainMap to store the environment frames, which allows us to push and pop
            # environments without copying the interpreter instance when we evaluate a jaxpr of
            # a dynamic decomposition. The name is different from the _env in the parent class
            # (a dictionary) to avoid confusion.
            self._env_map = ChainMap()

            gate_set, stopping_condition = _resolve_gate_set(gate_set, stopping_condition)
            self._gate_set = gate_set
            self._stopping_condition = stopping_condition

        def setup(self) -> None:
            """Setup the environment for the interpreter by pushing a new environment frame."""

            # This is the local environment for the jaxpr evaluation, on the top of the stack,
            # from which the interpreter reads and writes variables.
            # ChainMap writes to the first dictionary in the chain by default.
            self._env_map = self._env_map.new_child()

        def cleanup(self) -> None:
            """Cleanup the environment by popping the top-most environment frame."""

            # We delete the top-most environment frame after the evaluation is done.
            self._env_map = self._env_map.parents

        def read(self, var):
            """Extract the value corresponding to a variable."""
            return var.val if isinstance(var, jax.extend.core.Literal) else self._env_map[var]

        def stopping_condition(self, op: qml.operation.Operator) -> bool:
            """Function to determine whether an operator needs to be decomposed or not.

            Args:
                op (qml.operation.Operator): Operator to check.

            Returns:
                bool: Whether ``op`` is valid or needs to be decomposed. ``True`` means
                    that the operator does not need to be decomposed.
            """

            # If the new graph-based decomposition is enabled,
            # we don't rely on the has_decomposition attribute.
            if qml.decomposition.enabled_graph():
                return self._stopping_condition(op)

            if not op.has_decomposition:
                if not self._stopping_condition(op):
                    warnings.warn(
                        f"Operator {op.name} does not define a decomposition and was not "
                        f"found in the target gate set. To remove this warning, add the operator "
                        f"name ({op.name}) or type ({type(op)}) to the gate set.",
                        UserWarning,
                    )
                return True

            return self._stopping_condition(op)

        def decompose_operation(self, op: qml.operation.Operator):
            """Decompose a PennyLane operation instance if it does not satisfy the
            provided gate set.

            Args:
                op (Operator): a pennylane operator instance

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`, :meth:`~.interpret_operation`.
            """

            if self._stopping_condition(op):
                return self.interpret_operation(op)

            max_expansion = (
                self.max_expansion - self._current_depth if self.max_expansion is not None else None
            )

            with qml.capture.pause():
                decomposition = list(
                    _operator_decomposition_gen(
                        op,
                        self.stopping_condition,
                        max_expansion=max_expansion,
                        decomp_graph=self._decomp_graph,
                    )
                )

            return [self.interpret_operation(decomp_op) for decomp_op in decomposition]

        def _evaluate_jaxpr_decomposition(self, op: qml.operation.Operator):
            """Creates and evaluates a Jaxpr of the plxpr decomposition of an operator."""

            if self._stopping_condition(op):
                return self.interpret_operation(op)

            if self.max_expansion is not None and self._current_depth >= self.max_expansion:
                return self.interpret_operation(op)

            if qml.decomposition.enabled_graph() and self._decomp_graph.is_solved_for(op):

                rule = self._decomp_graph.decomposition(op)
                num_wires = len(op.wires)

                def compute_qfunc_decomposition(*_args, **_kwargs):
                    wires = qml.math.array(_args[-num_wires:], like="jax")
                    rule(*_args[:-num_wires], wires=wires, **_kwargs)

            else:
                compute_qfunc_decomposition = op.compute_qfunc_decomposition

            args = (*op.parameters, *op.wires)

            jaxpr_decomp = qml.capture.make_plxpr(
                partial(compute_qfunc_decomposition, **op.hyperparameters)
            )(*args)

            self._current_depth += 1
            # We don't need to copy the interpreter here, as the jaxpr of the decomposition
            # is evaluated with a new environment frame placed on top of the stack.
            out = self.eval(jaxpr_decomp.jaxpr, jaxpr_decomp.consts, *args)
            self._current_depth -= 1

            return out

        # pylint: disable=too-many-branches
        def eval(self, jaxpr: jax.extend.core.Jaxpr, consts: Sequence, *args) -> list:
            """
            Evaluates a jaxpr, which can also be generated by a dynamic decomposition.

            Args:
                jaxpr_decomp (jax.extend.core.Jaxpr): the Jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args: the arguments to use in the evaluation
            """

            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env_map[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env_map[constvar] = const

            if qml.decomposition.enabled_graph() and not self._decomp_graph:

                with qml.capture.pause():

                    collector = CollectResourceOps()
                    collector.eval(jaxpr, consts, *args)
                    operations = collector.state["ops"]

                if operations:
                    self._decomp_graph = _construct_and_solve_decomp_graph(
                        operations,
                        self._gate_set,
                        self._fixed_decomps,
                        self._alt_decomps,
                    )

            for eq in jaxpr.eqns:

                prim_type = getattr(eq.primitive, "prim_type", "")
                custom_handler = self._primitive_registrations.get(eq.primitive, None)

                if custom_handler:

                    invals = [self.read(invar) for invar in eq.invars]
                    outvals = custom_handler(self, *invals, **eq.params)

                elif prim_type == "operator":
                    outvals = self.interpret_operation_eqn(eq)
                elif prim_type == "measurement":
                    outvals = self.interpret_measurement_eqn(eq)
                else:
                    invals = [self.read(invar) for invar in eq.invars]
                    subfuns, params = eq.primitive.get_bind_params(eq.params)
                    outvals = eq.primitive.bind(*subfuns, *invals, **params)

                if not eq.primitive.multiple_results:
                    outvals = [outvals]

                for outvar, outval in zip(eq.outvars, outvals, strict=True):
                    self._env_map[outvar] = outval

            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, qml.operation.Operator):
                    outvals.append(self.interpret_operation(outval))
                else:
                    outvals.append(outval)

            self.cleanup()

            return outvals

        def interpret_operation_eqn(self, eqn: jax.extend.core.JaxprEqn):
            """Interpret an equation corresponding to an operator.

            If the operator has a dynamic decomposition defined, this method will
            create and evaluate the jaxpr of the decomposition using the :meth:`~.eval` method.

            Args:
                eqn (jax.extend.core.JaxprEqn): a jax equation for an operator.

            See also: :meth:`~.interpret_operation`.

            """

            invals = (self.read(invar) for invar in eqn.invars)

            with qml.QueuingManager.stop_recording():
                op = eqn.primitive.impl(*invals, **eqn.params)

            if not eqn.outvars[0].__class__.__name__ == "DropVar":
                return op

            # _evaluate_jaxpr_decomposition should be used when the operator defines a
            # compute_qfunc_decomposition, or if graph-based decomposition is enabled and
            # a solution is found for this operator in the graph.
            if (
                op.has_qfunc_decomposition
                or qml.decomposition.enabled_graph()
                and self._decomp_graph.is_solved_for(op)
            ):
                return self._evaluate_jaxpr_decomposition(op)

            return self.decompose_operation(op)

    # pylint: disable=too-many-arguments
    @DecomposeInterpreter.register_primitive(ctrl_transform_prim)
    def _(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
        consts = invals[:n_consts]
        args = invals[n_consts:-n_control]
        control_wires = invals[-n_control:]

        unroller = ControlTransformInterpreter(
            control_wires, control_values=control_values, work_wires=work_wires
        )

        def wrapper(*inner_args):
            return unroller.eval(jaxpr, consts, *inner_args)

        jaxpr = jax.make_jaxpr(wrapper)(*args)
        return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)

    def decompose_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args):
        """Function for applying the ``decompose`` transform on plxpr."""

        interpreter = DecomposeInterpreter(*targs, **tkwargs)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return jax.make_jaxpr(wrapper)(*args)

    return DecomposeInterpreter, decompose_plxpr_to_plxpr


DecomposeInterpreter, decompose_plxpr_to_plxpr = _get_plxpr_decompose()


@partial(transform, plxpr_transform=decompose_plxpr_to_plxpr)
def decompose(
    tape,
    *,
    gate_set=None,
    stopping_condition=None,
    max_expansion=None,
    fixed_decomps: dict = None,
    alt_decomps: dict = None,
):  # pylint: disable=too-many-arguments
    """Decomposes a quantum circuit into a user-specified gate set.

    .. note::

        When ``qml.decomposition.enable_graph()`` is present, this transform takes advantage of the
        new graph-based decomposition algorithm that allows for more flexible and resource-efficient
        decompositions towards any target gate set. The keyword arguments ``fixed_decomps`` and
        ``alt_decomps`` are only functional with this toggle present.

    .. seealso::

        For more information on PennyLane's decomposition tools and features, check out the
        :doc:`Compiling Circuits page </introduction/compiling_circuits>`.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_set (Iterable[str or type], Dict[type or str, float] or Callable, optional): The
            target gate set specified as either (1) a sequence of operator types and/or names,
            (2) a dictionary mapping operator types and/or names to their respective costs, in
            which case the total cost will be minimized (only available when the new graph-based
            decomposition system is enabled), or (3) a function that returns ``True`` if the
            operator belongs to the target gate set (not supported with the new graph-based
            decomposition system). If ``None``, the gate set is considered to be all available
            :doc:`quantum operators </introduction/operations>`.
        stopping_condition (Callable, optional): a function that returns ``True`` if the operator
            does not need to be decomposed. If ``None``, the default stopping condition is whether
            the operator is in the target gate set. See the "Gate Set vs. Stopping Condition"
            section below for more details.
        max_expansion (int, optional): The maximum depth of the decomposition. Defaults to ``None``.
            If ``None``, the circuit will be decomposed until the target gate set is reached.
        fixed_decomps (Dict[Type[Operator], DecompositionRule]): a dictionary mapping operator types
            to custom decomposition rules. A decomposition rule is a quantum function decorated with
            :func:`~pennylane.register_resources`. The custom decomposition rules specified here
            will be used in place of the existing decomposition rules defined for this operator.
            This is only used when :func:`~pennylane.decomposition.enable_graph` is present.
        alt_decomps (Dict[Type[Operator], List[DecompositionRule]]): a dictionary mapping operator
            types to lists of alternative custom decomposition rules. A decomposition rule is a
            quantum function decorated with :func:`~pennylane.register_resources`. The custom
            decomposition rules specified here will be considered as alternatives to the existing
            decomposition rules defined for this operator, and one of them may be chosen if they
            lead to a more resource-efficient decomposition. This is only used when :func:`~pennylane.decomposition.enable_graph`
            is present.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    .. note::

        This function does not guarantee a decomposition to the target gate set. If an operation
        with no defined decomposition is encountered during decomposition, it will be left in the
        circuit even if it does not belong in the target gate set. In this case, a ``UserWarning``
        will be raised. To suppress this warning, simply add the operator to the gate set.
        When ``qml.decomposition.enabled_graph()``, PennyLane errors out with a ``DecompositionError``.

    .. seealso::

        For decomposing into Clifford + T, check out :func:`~.pennylane.clifford_t_decomposition`.

        :func:`qml.devices.preprocess.decompose <.pennylane.devices.preprocess.decompose>` is a
        transform that is intended for device developers. This function will decompose a quantum
        circuit into a set of basis gates available on a specific device architecture.

    **Example**

    Consider the following tape:

    >>> ops = [qml.IsingXX(1.2, wires=(0,1))]
    >>> tape = qml.tape.QuantumScript(ops, measurements=[qml.expval(qml.Z(0))])

    You can decompose the circuit into a set of gates:

    >>> batch, fn = qml.transforms.decompose(tape, gate_set={qml.CNOT, qml.RX})
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]), RX(1.2, wires=[0]), CNOT(wires=[0, 1]), expval(Z(0))]

    You can also apply the transform directly on a :class:`~.pennylane.QNode`:

    .. code-block:: python

        from functools import partial

        @partial(qml.transforms.decompose, gate_set={qml.Toffoli, "RX", "RZ"})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(wires=[0])
            qml.Toffoli(wires=[0,1,2])
            return qml.expval(qml.Z(0))

    Since the Hadamard gate is not defined in our gate set, it will be decomposed into rotations:

    >>> print(qml.draw(circuit)())
    0: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭●─┤  <Z>
    1: ───────────────────────────────├●─┤
    2: ───────────────────────────────╰X─┤

    You can also use a function to build a decomposition gate set:

    .. code-block:: python

        @partial(qml.transforms.decompose, gate_set=lambda op: len(op.wires)<=2)
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(wires=[0])
            qml.Toffoli(wires=[0,1,2])
            return qml.expval(qml.Z(0))

    The circuit will be decomposed into single or two-qubit operators,

    >>> print(qml.draw(circuit)())
    0: ──H────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

    You can use the ``max_expansion`` argument to control the number of decomposition stages
    applied to the circuit. By default, the function will decompose the circuit until the desired
    gate set is reached.

    The example below demonstrates how the user can visualize the decomposition in stages:

    .. code-block:: python

        phase = 1
        target_wires = [0]
        unitary = qml.RX(phase, wires=0).matrix()
        n_estimation_wires = 3
        estimation_wires = range(1, n_estimation_wires + 1)

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            # Start in the |+> eigenstate of the unitary
            qml.Hadamard(wires=target_wires)
            qml.QuantumPhaseEstimation(
                unitary,
                target_wires=target_wires,
                estimation_wires=estimation_wires,
            )

    >>> print(qml.draw(qml.transforms.decompose(circuit, max_expansion=0))())
    0: ──H─╭QuantumPhaseEstimation─┤
    1: ────├QuantumPhaseEstimation─┤
    2: ────├QuantumPhaseEstimation─┤
    3: ────╰QuantumPhaseEstimation─┤

    >>> print(qml.draw(qml.transforms.decompose(circuit, max_expansion=1))())
    0: ──H─╭U(M0)⁴─╭U(M0)²─╭U(M0)¹───────┤
    1: ──H─╰●──────│───────│───────╭QFT†─┤
    2: ──H─────────╰●──────│───────├QFT†─┤
    3: ──H─────────────────╰●──────╰QFT†─┤

    >>> print(qml.draw(qml.transforms.decompose(circuit, max_expansion=2))())
    0: ──H──RZ(4.71)──RY(1.14)─╭X──RY(-1.14)──RZ(-3.14)─╭X──RZ(-1.57)──RZ(1.57)──RY(1.00)─╭X ···
    1: ──H─────────────────────╰●───────────────────────╰●────────────────────────────────│─ ···
    2: ──H────────────────────────────────────────────────────────────────────────────────╰● ···
    3: ──H────────────────────────────────────────────────────────────────────────────────── ···
    <BLANKLINE>
    0: ··· ──RY(-1.00)──RZ(-6.28)─╭X──RZ(4.71)──RZ(1.57)──RY(0.50)─╭X──RY(-0.50)──RZ(-6.28)─╭X ···
    1: ··· ───────────────────────│────────────────────────────────│────────────────────────│─ ···
    2: ··· ───────────────────────╰●───────────────────────────────│────────────────────────│─ ···
    3: ··· ────────────────────────────────────────────────────────╰●───────────────────────╰● ···
    <BLANKLINE>
    0: ··· ──RZ(4.71)────────────────────────────────────────────────────┤
    1: ··· ─╭SWAP†─────────────────────────╭(Rϕ(0.79))†─╭(Rϕ(1.57))†──H†─┤
    2: ··· ─│─────────────╭(Rϕ(1.57))†──H†─│────────────╰(Rϕ(1.57))†─────┤
    3: ··· ─╰SWAP†─────H†─╰(Rϕ(1.57))†─────╰(Rϕ(0.79))†──────────────────┤

    .. details::
        :title: Integration with the Graph-Based Decomposition System

        This transform takes advantage of the new graph-based decomposition algorithm when
        ``qml.decomposition.enable_graph()`` is present, which allows for more flexible
        decompositions towards any target gate set. For example, the current system does not
        guarantee a decomposition to the desired target gate set:

        .. code-block:: python

            import pennylane as qml

            with qml.queuing.AnnotatedQueue() as q:
                qml.CRX(0.5, wires=[0, 1])

            tape = qml.tape.QuantumScript.from_queue(q)
            [new_tape], _ = qml.transforms.decompose([tape], gate_set={"RX", "RY", "RZ", "CZ"})

        .. code-block:: pycon

            >>> new_tape.operations
            [RZ(1.5707963267948966, wires=[1]),
             RY(0.25, wires=[1]),
             CNOT(wires=[0, 1]),
             RY(-0.25, wires=[1]),
             CNOT(wires=[0, 1]),
             RZ(-1.5707963267948966, wires=[1])]

        With the new system enabled, the transform produces the expected outcome.

        .. code-block:: pycon

            >>> qml.decomposition.enable_graph()
            >>> [new_tape], _ = qml.transforms.decompose([tape], gate_set={"RX", "RY", "RZ", "CZ"})
            >>> new_tape.operations
            [RX(0.25, wires=[1]), CZ(wires=[0, 1]), RX(-0.25, wires=[1]), CZ(wires=[0, 1])]

        **Weighted Gate Sets**

        With the graph based decomposition enabled, gate weights can be provided in the ``gate_set`` parameter. For example:

        .. code-block:: python

            @partial(
                qml.transforms.decompose,
                gate_set={qml.Toffoli: 1.23, qml.RX: 4.56, qml.CZ: 0.01, qml.H: 420, qml.CRZ: 100}
            )
            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                qml.CRX(0.1, wires=[0, 1])
                qml.Toffoli(wires=[0, 1, 2])
                return qml.expval(qml.Z(0))

        .. code-block:: pycon

            >>> print(qml.draw(circuit)())
            0: ───────────╭●────────────╭●─╭●─┤  <Z>
            1: ──RX(0.05)─╰Z──RX(-0.05)─╰Z─├●─┤
            2: ────────────────────────────╰X─┤

        .. code-block:: python

            @partial(
                qml.transforms.decompose,
                gate_set={qml.Toffoli: 1.23, qml.RX: 4.56, qml.CZ: 0.01, qml.H: 0.1, qml.CRZ: 0.1}
            )
            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                qml.CRX(0.1, wires=[0, 1])
                qml.Toffoli(wires=[0, 1, 2])
                return qml.expval(qml.Z(0))

        .. code-block:: pycon

            >>> print(qml.draw(circuit)())
            0: ────╭●───────────╭●─┤  <Z>
            1: ──H─╰RZ(0.10)──H─├●─┤
            2: ─────────────────╰X─┤


        Here, when the Hadamard and ``CRZ`` have relatively high weights, a decomposition involving them is considered
        *less* efficient. When they have relatively low weights, a decomposition involving them is considered *more*
        efficient.

        **Gate Set vs. Stopping Condition**

        With the new graph-based decomposition system enabled, we make the distinction between a
        target gate set and a stopping condition. The ``gate_set`` is a collection of operator
        types and/or names that is required by the graph-based decomposition solver, which chooses
        a decomposition rule for each operator that ultimately minimizes the total number of gates
        in terms of the target gate set (or the total cost if weights are provided). On the other
        hand, the ``stopping_condition`` is a function that determines whether an operator instance
        needs to be decomposed. In short, the ``gate_set`` is specified in terms of operator types,
        whereas the ``stopping_condition`` is specified in terms of operator instances.

        Here is an example of using ``stopping_condition`` to not decompose a ``qml.QubitUnitary``
        instance if it's equivalent to the identity matrix.

        .. code-block:: python

            from functools import partial
            import pennylane as qml

            qml.decomposition.enable_graph()

            # Prepare a unitary matrix that we want to decompose
            U = qml.matrix(qml.Rot(0.1, 0.2, 0.3, wires=0) @ qml.Identity(wires=1))

            def stopping_condition(op):

                if isinstance(op, qml.QubitUnitary):
                    identity = qml.math.eye(2 ** len(op.wires))
                    return qml.math.allclose(op.matrix(), identity)

                return False

        Note that the ``stopping_condition`` does not need to check whether the operator is in the
        target gate set. This will always be checked.

        .. code-block:: python

            @partial(
                qml.transforms.decompose,
                gate_set={qml.RZ, qml.RY, qml.GlobalPhase, qml.CNOT},
                stopping_condition=stopping_condition,
            )
            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                qml.QubitUnitary(U, wires=[0, 1])
                return qml.expval(qml.PauliZ(0))

        .. code-block:: pycon

            >>> print(qml.draw(circuit)())
            0: ──RZ(0.10)──RY(0.20)──RZ(0.30)─┤  <Z>
            1: ──U(M0)────────────────────────┤

            M0 =
            [[1.+0.j 0.+0.j]
             [0.+0.j 1.+0.j]]

        We can see that the ``QubitUnitary`` on wire 1 is not decomposed due to the stopping
        condition, despite ``QubitUnitary`` not being in the target gate set.

        **Customizing Decompositions**

        The new system also enables specifying custom decomposition rules. When ``qml.decomposition.enable_graph()``
        is present, this transform accepts two additional keyword arguments: ``fixed_decomps`` and
        ``alt_decomps``. The user can define custom decomposition rules as quantum functions decorated
        with ``@qml.register_resources``, and provide them to the transform via these arguments.

        .. seealso:: :func:`qml.register_resources <pennylane.register_resources>`

        The ``fixed_decomps`` forces the transform to use the specified decomposition rules for
        certain operators, whereas the ``alt_decomps`` is used to provide alternative decomposition rules
        for operators that may be chosen if they lead to a more resource-efficient decomposition.

        In the following example, ``isingxx_decomp`` will always be used to decompose ``qml.IsingXX``
        gates; when it comes to ``qml.CNOT``, the system will choose the most efficient decomposition rule
        among ``my_cnot1``, ``my_cnot2``, and all existing decomposition rules defined for ``qml.CNOT``.

        .. code-block:: python

            import pennylane as qml

            qml.decomposition.enable_graph()

            @qml.register_resources({qml.CNOT: 2, qml.RX: 1})
            def isingxx_decomp(phi, wires, **__):
                qml.CNOT(wires=wires)
                qml.RX(phi, wires=[wires[0]])
                qml.CNOT(wires=wires)

            @qml.register_resources({qml.H: 2, qml.CZ: 1})
            def my_cnot1(wires, **__):
                qml.H(wires=wires[1])
                qml.CZ(wires=wires)
                qml.H(wires=wires[1])

            @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
            def my_cnot2(wires, **__):
                qml.RY(np.pi/2, wires[1])
                qml.Z(wires[1])
                qml.CZ(wires=wires)
                qml.RY(np.pi/2, wires[1])
                qml.Z(wires[1])

            @partial(
                qml.transforms.decompose,
                gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
                alt_decomps={qml.CNOT: [my_cnot1, my_cnot2]},
                fixed_decomps={qml.IsingXX: isingxx_decomp},
            )
            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                qml.CNOT(wires=[0, 1])
                qml.IsingXX(0.5, wires=[0, 1])
                return qml.state()


        .. code-block:: pycon

            >>> qml.specs(circuit)()["resources"].gate_types
            defaultdict(int, {'RZ': 12, 'RX': 7, 'GlobalPhase': 6, 'CZ': 3})

    """

    if not qml.decomposition.enabled_graph() and (fixed_decomps or alt_decomps):
        raise TypeError(
            "The keyword arguments fixed_decomps and alt_decomps are only available with the new "
            "experimental graph-based decomposition system. Use qml.decomposition.enable_graph() "
            "to enable the new system."
        )

    gate_set, stopping_condition = _resolve_gate_set(gate_set, stopping_condition)

    def _stopping_condition(op):

        # If the new graph-based decomposition is enabled,
        # we don't rely on the has_decomposition attribute.
        if qml.decomposition.enabled_graph():
            return stopping_condition(op)

        if not op.has_decomposition:
            if not stopping_condition(op):
                warnings.warn(
                    f"Operator {op.name} does not define a decomposition and was not "
                    f"found in the target gate set. To remove this warning, add the operator name "
                    f"({op.name}) or type ({type(op)}) to the gate set.",
                    UserWarning,
                )
            return True

        return stopping_condition(op)

    if all(_stopping_condition(op) for op in tape.operations):
        return (tape,), null_postprocessing

    # If the decomposition graph is enabled, we create a DecompositionGraph instance
    # to optimize the decomposition.
    decomp_graph = None

    if qml.decomposition.enabled_graph():

        decomp_graph = _construct_and_solve_decomp_graph(
            tape.operations,
            gate_set,
            fixed_decomps=fixed_decomps,
            alt_decomps=alt_decomps,
        )

    try:
        new_ops = [
            final_op
            for op in tape.operations
            for final_op in _operator_decomposition_gen(
                op, _stopping_condition, max_expansion=max_expansion, decomp_graph=decomp_graph
            )
        ]
    except RecursionError as e:
        raise RecursionError(
            "Reached recursion limit trying to decompose operations. Operator decomposition may "
            "have entered an infinite loop. Setting max_expansion will terminate the decomposition "
            "at a fixed recursion depth."
        ) from e

    tape = tape.copy(operations=new_ops)

    return (tape,), null_postprocessing


def _operator_decomposition_gen(
    op: qml.operation.Operator,
    acceptance_function: Callable[[qml.operation.Operator], bool],
    max_expansion: int | None = None,
    current_depth=0,
    decomp_graph: DecompositionGraph = None,
) -> Generator[qml.operation.Operator]:
    """A generator that yields the next operation that is accepted."""

    max_depth_reached = False
    decomp = []

    if max_expansion is not None and max_expansion <= current_depth:
        max_depth_reached = True

    if acceptance_function(op) or max_depth_reached:
        yield op
    elif decomp_graph is not None and decomp_graph.is_solved_for(op):
        op_rule = decomp_graph.decomposition(op)
        with qml.queuing.AnnotatedQueue() as decomposed_ops:
            op_rule(*op.parameters, wires=op.wires, **op.hyperparameters)
        decomp = decomposed_ops.queue
        current_depth += 1
    else:
        decomp = op.decomposition()
        current_depth += 1

    for sub_op in decomp:
        yield from _operator_decomposition_gen(
            sub_op,
            acceptance_function,
            max_expansion=max_expansion,
            current_depth=current_depth,
            decomp_graph=decomp_graph,
        )


def _resolve_gate_set(
    gate_set: set[type | str] | dict[type | str, float] = None,
    stopping_condition: Callable[[qml.operation.Operator], bool] = None,
) -> tuple[set[type | str] | dict[type | str, float], Callable[[qml.operation.Operator], bool]]:
    """Resolve the gate set and the stopping condition from arguments.

    The ``gate_set`` can be provided in various forms, and the ``stopping_condition`` may or
    may not be provided. This function will resolve the gate set and the stopping condition
    to the following standardized form:

    - The ``gate_set`` is set of operator **types** and/or names, or a dictionary mapping operator
      types and/or names to their respective costs. This is only used by the DecompositionGraph
    - The ``stopping_condition`` is a function that takes an operator **instances** and returns
      ``True`` if the operator does not need to be decomposed. This is used during decomposition.

    """

    if not qml.decomposition.enabled_graph() and gate_set and stopping_condition:
        # In the old decomposition system, we don't make the distinction between a set of
        # target gates and the stopping condition, so to avoid ambiguity, we don't allow both
        # to be specified at the same time.
        raise TypeError(
            "Specifying both gate_set and stopping_condition is only supported with the new "
            "experimental graph-based decomposition system. Use qml.decomposition.enable_graph() "
            "to enable the new system."
        )

    if gate_set is None:
        gate_set = set(qml.ops.__all__)

    if isinstance(gate_set, (str, type)):
        gate_set = {gate_set}

    if isinstance(gate_set, dict):

        if any(v < 0 for v in gate_set.values()):
            raise ValueError("Negative gate weights provided to gate_set are not supported.")

        if not qml.decomposition.enabled_graph():
            raise TypeError(
                "Specifying the gate_set with a dictionary of operator types and their weights "
                "is only supported with the new experimental graph-based decomposition system. "
                "Enable the new system using qml.decomposition.enable_graph()."
            )

    if isinstance(gate_set, Iterable):

        gate_types = tuple(gate for gate in gate_set if isinstance(gate, type))
        gate_names = {translate_op_alias(gate) for gate in gate_set if isinstance(gate, str)}

        def gate_set_contains(op: Operator) -> bool:
            return (op.name in gate_names) or isinstance(op, gate_types)

    elif isinstance(gate_set, Callable):  # pylint:disable=isinstance-second-argument-not-valid-type

        gate_set_contains = gate_set

        if qml.decomposition.enabled_graph():
            raise TypeError(
                "Specifying gate_set as a function is not supported with the new "
                "graph-based decomposition system enabled."
            )

    else:
        raise TypeError("Invalid gate_set type. Must be an iterable, dictionary, or function.")

    if stopping_condition:

        # Even when the user provides a stopping condition, we still need to check
        # whether an operator belongs to the target gate set. This is to prevent
        # the case of an operator missing the stopping condition but doesn't have
        # a decomposition assigned due to being in the target gate set.
        def _stopping_condition(op):
            return gate_set_contains(op) or stopping_condition(op)

    else:
        # If the stopping condition is not explicitly provided, the default is to simply check
        # whether an operator belongs to the target gate set.
        _stopping_condition = gate_set_contains

    return gate_set, _stopping_condition


def _construct_and_solve_decomp_graph(operations, target_gates, fixed_decomps, alt_decomps):
    """Create and solve a DecompositionGraph instance to optimize the decomposition."""

    # Create the decomposition graph
    decomp_graph = DecompositionGraph(
        operations,
        target_gates,
        fixed_decomps=fixed_decomps,
        alt_decomps=alt_decomps,
    )

    # Find the efficient pathways to the target gate set
    decomp_graph.solve()
    return decomp_graph
