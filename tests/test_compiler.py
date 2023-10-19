# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the compiler subpackage.
"""

import pytest

import pennylane as qml

catalyst = pytest.mark.external("catalyst")
jax = pytest.mark.external("jax")

from jax import numpy as jnp  # pylint:disable=wrong-import-position
from jax.core import ShapedArray  # pylint:disable=wrong-import-position

# pylint: disable=too-few-public-methods, too-many-public-methods


class TestCatalyst:
    """Test ``qml.qjit`` with Catalyst"""

    def test_compiler(self):
        """Test compiler active and available methods"""

        assert not qml.compiler.active()

        assert qml.compiler.available("catalyst")
        assert qml.compiler.available_compilers() == ["catalyst"]

        assert qml.compiler.available("catalyst")
        assert qml.compiler.available_compilers() == ["catalyst"]

    def test_qjit_circuit(self):
        """Test JIT compilation of a circuit with 2-qubit"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=1))

        assert jnp.allclose(circuit(0.5), 0.0)

    def test_qjit_aot(self):
        """Test AOT compilation of a circuit with 2-qubit"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit  # compilation happens at definition
        @qml.qnode(dev)
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

        result = circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))
        expected = jnp.array(
            [0.75634905 - 0.52801002j, 0.0 + 0.0j, 0.35962678 + 0.14074839j, 0.0 + 0.0j]
        )
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize(
        "_in,_out",
        [
            (0, False),
            (1, True),
        ],
    )
    def test_variable_capture_multiple_devices(self, _in, _out):
        """Test variable capture using multiple backend devices."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit()
        def workflow(n: int):
            @qml.qnode(dev)
            def f(x: float):
                qml.RX(n * x, wires=n)
                return qml.expval(qml.PauliZ(wires=n))

            @qml.qnode(dev)
            def g(x: float):
                qml.RX(x, wires=1)
                return qml.expval(qml.PauliZ(wires=1))

            return jnp.array_equal(f(jnp.pi), g(jnp.pi))

        assert workflow(_in) == _out

    def test_args_workflow(self):
        """Test arguments with workflows."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        def workflow1(params1, params2):
            """A classical workflow"""
            res1 = params1["a"][0][0] + params2[1]
            return jnp.sin(res1)

        params1 = {
            "a": [[0.1], 0.2],
        }
        params2 = (0.6, 0.8)
        expected = 0.78332691
        result = workflow1(params1, params2)
        assert jnp.allclose(result, expected)

    def test_return_value_dict(self):
        """Test pytree return values."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return {
                "w0": qml.expval(qml.PauliZ(0)),
                "w1": qml.expval(qml.PauliZ(1)),
            }

        jitted_fn = qml.qjit(circuit1)

        params = [0.2, 0.6]
        expected = {"w0": 0.98006658, "w1": 0.82533561}
        result = jitted_fn(params)
        assert isinstance(result, dict)
        assert jnp.allclose(result["w0"], expected["w0"])
        assert jnp.allclose(result["w1"], expected["w1"])

    def test_qjit_python_if(self):
        """Test JIT compilation with the autograph support"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(autograph=True)
        @qml.qnode(dev)
        def circuit(x: int):
            if x < 5:
                qml.Hadamard(wires=0)
            else:
                qml.T(wires=0)

            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(3), 0.0)
        assert jnp.allclose(circuit(5), 1.0)

    def test_compilation_opt(self):
        """Test user-configurable compilation options"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir")
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x**2, wires=1)
            return qml.expval(qml.PauliZ(0))

        mlir_str = str(circuit.mlir)
        result_header = "func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64>"
        assert result_header in mlir_str
