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
Tests for the Reflection Operator template
"""

import pytest
import numpy as np
import pennylane as qml


@qml.prod
def hadamards(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


@pytest.mark.parametrize(
    ("prod", "reflection_wires"),
    [
        (qml.QFT([0, 1, 4]), [0, 1, 2]),
        (qml.QFT([0, 1, 2]), [3]),
        (qml.QFT([0, 1, 2]), [0, 1, 2, 3]),
    ],
)
def test_reflection_wires(prod, reflection_wires):
    """Assert reflection_wires is a subset of the U wires"""
    with pytest.raises(
        ValueError, match="The reflection wires must be a subset of the operation wires."
    ):
        qml.Reflection(prod, 0.5, reflection_wires=reflection_wires)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (
            qml.Reflection(qml.Hadamard(wires=0), 0.5, reflection_wires=[0]),
            [
                qml.GlobalPhase(np.pi),
                qml.adjoint(qml.Hadamard(0)),
                qml.PauliX(wires=[0]),
                qml.PhaseShift(0.5, wires=[0]),
                qml.PauliX(wires=[0]),
                qml.Hadamard(0),
            ],
        ),
        (
            qml.Reflection(qml.QFT(wires=[0, 1]), 0.5),
            [
                qml.GlobalPhase(np.pi),
                qml.adjoint(qml.QFT(wires=[0, 1])),
                qml.PauliX(wires=[1]),
                qml.ctrl(qml.PhaseShift(0.5, wires=[1]), control=0, control_values=[0]),
                qml.PauliX(wires=[1]),
                qml.QFT(wires=[0, 1]),
            ],
        ),
    ],
)
def test_decomposition(op, expected):
    """Test that the decomposition of the Reflection operator is correct"""
    decomp = op.decomposition()
    assert decomp == expected


def test_default_values():
    """Test that the default values are correct"""

    U = qml.QFT(wires=[0, 1, 4])
    op = qml.Reflection(U)

    assert op.alpha == np.pi
    assert op.reflection_wires == U.wires


@pytest.mark.parametrize("n_wires", [3, 4, 5])
def test_grover_as_reflection(n_wires):
    """Test that the GroverOperator can be used as a Reflection operator"""

    grover_matrix = qml.matrix(qml.GroverOperator(wires=range(n_wires)))
    reflection_matrix = qml.matrix(qml.Reflection(hadamards(wires=range(n_wires))))

    assert np.allclose(grover_matrix, reflection_matrix)


class TestIntegration:
    """Tests that the Reflection is executable and differentiable in a QNode context"""

    @staticmethod
    def circuit(alpha):
        """Test circuit"""
        qml.RY(1.2, wires=0)
        qml.RY(-1.4, wires=1)
        qml.RX(-2, wires=0)
        qml.CRX(1, wires=[0, 1])
        qml.Reflection(hadamards(range(3)), alpha)
        return qml.expval(qml.PauliZ(0))

    x = np.array(0.25)

    # not calculated analytically, we are only ensuring that the results are consistent accross interfaces

    exp_result = np.array(
        [
            2.48209280e-01,
            2.79302360e-05,
            1.76417324e-01,
            2.79302360e-05,
            3.18107276e-01,
            2.79302360e-05,
            2.57154398e-01,
            2.79302360e-05,
        ]
    )
    exp_jac = np.array(
        [
            -0.00322306,
            0.00022228,
            0.00312425,
            0.00022228,
            0.01377867,
            0.00022228,
            -0.01456896,
            0.00022228,
        ]
    )

    def test_qnode_numpy(self):
        """Test that the QNode executes with Numpy."""
        dev = qml.device("default.qubit")
        qnode = qml.QNode(self.circuit, dev, interface=None)

        res = qnode(self.x)
        assert res.shape == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    def test_lightning_qubit(self):
        """Test that the QNode executes with the Lightning Qubit simulator."""
        dev = qml.device("lightning.qubit", wires=3)
        qnode = qml.QNode(self.circuit, dev)

        res = qnode(self.x)
        assert res.shape == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    @pytest.mark.autograd
    def test_qnode_autograd(self):
        """Test that the QNode executes with Autograd."""

        dev = qml.device("default.qubit")
        qnode = qml.QNode(self.circuit, dev, interface="autograd")

        x = qml.numpy.array(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("shots", [None, 10000])
    def test_qnode_jax(self, shots, use_jit):
        """Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", shots=shots, seed=10)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method)
        if use_jit:
            qnode = jax.jit(qnode)

        x = jax.numpy.array(self.x)
        res = qnode(x)
        assert qml.math.shape(res) == (16,)
        assert np.allclose(res, self.exp_result, atol=0.005)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(x)
        assert jac.shape == (16,)
        assert np.allclose(jac, self.exp_jac, atol=0.006)

    @pytest.mark.torch
    @pytest.mark.parametrize("shots", [None, 10000])
    def test_qnode_torch(self, shots):
        """Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import torch

        dev = qml.device("default.qubit", shots=shots, seed=10)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method)

        x = torch.tensor(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == (16,)
        assert qml.math.allclose(res, self.exp_result, atol=0.002)

        jac = torch.autograd.functional.jacobian(qnode, x)
        assert qml.math.shape(jac) == (16,)
        assert qml.math.allclose(jac, self.exp_jac, atol=0.006)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots", [None, 10000])
    @pytest.mark.xfail(reason="tf gradient doesn't seem to be working, returns ()")
    def test_qnode_tf(self, shots):
        """Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = qml.device("default.qubit", shots=shots, seed=10)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method)

        x = tf.Variable(self.x)
        with tf.GradientTape() as tape:
            res = qnode(x)

        assert qml.math.shape(res) == (16,)
        assert qml.math.allclose(res, self.exp_result, atol=0.002)

        jac = tape.gradient(res, x)
        assert qml.math.shape(jac) == (16,)


def test_correct_queueing():
    """Test that the Reflection operator is correctly queued in the circuit"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit1():
        qml.Hadamard(wires=0)
        qml.RY(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=qml.Hadamard(wires=0), alpha=2.0)
        return qml.state()

    @qml.prod
    def generator(wires):
        qml.Hadamard(wires=wires)

    @qml.qnode(dev)
    def circuit2():
        generator(wires=0)
        qml.RY(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=generator(wires=0), alpha=2.0)
        return qml.state()

    U = generator(0)

    @qml.qnode(dev)
    def circuit3():
        generator(wires=0)
        qml.RY(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=U, alpha=2.0)
        return qml.state()

    assert np.allclose(circuit1(), circuit2())
    assert np.allclose(circuit1(), circuit3())
