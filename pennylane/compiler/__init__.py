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
r"""
.. currentmodule:: pennylane

This module provides support for hybrid quantum-classical compilation.
Through the use of the :func:`~.qjit` decorator, entire workflows
can be just-in-time (JIT) compiled --- including both quantum and
classical processing --- down to a machine binary on first
function execution. Subsequent calls to the compiled function will execute
previously compiled binary, resulting in significant
performance improvements.

Currently, PennyLane supports the
`Catalyst <https://github.com/pennylaneai/catalyst>`__ hybrid compiler
with the :func:`~.qjit` decorator. A significant benefit of Catalyst
is the ability to preserve complex control flow around quantum
operations — such as if statements and for loops, and including measurement feedforward — during compilation, while continuing to support end-to-end
autodifferentiation.

.. note::

    Catalyst currently only support the JAX interface of PennyLane.

Overview
--------

The main entry point to hybrid compilation in PennyLane
is via the qjit decorator.

.. autosummary::
    :toctree: api

    ~qjit

In addition, several developer functions are available to probe
available hybrid compilers.

.. autosummary::
    :toctree: api

    ~compiler.available_compilers
    ~compiler.available
    ~compiler.active

Compiler
--------
The compiler module provides the infrastructure to integrate external
hybrid quantum-classical compilers with PennyLane, but does not provide
a built-in compiler.

Currently, only the `Catalyst <https://github.com/pennylaneai/catalyst>`__
hybrid compiler is supported with PennyLane, however there are plans
to incorporate additional compilers in the near future.

.. note::

    Catalyst is officially supported on Linux (x86_64) and macOS (aarch64) platforms. To install it, simply run
    the following ``pip`` command:

    .. code-block:: console

      pip install pennylane-catalyst

    Please see the `installation <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/installation.html>`__ guide
    for more information.

For any compiler packages seeking to be registered, it is imperative that they expose the 'entry_points' metadata
under the designated group name: ``pennylane.compilers``.

Basic usage
-----------

When using just-in-time (JIT) compilation, the compilation is triggered at the call site the
first time the quantum function is executed. For example, ``circuit`` is
compiled as early as the first call.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(wires=1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    Alternatively, if argument type hints are provided, compilation
    can occur 'ahead of time' when the function is decorated.

    .. code-block:: python

        from jax.core import ShapedArray

        @qjit  # compilation happens at definition
        @qml.qnode(dev)
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

    >>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
    array([0.75634905-0.52801002j, 0. +0.j,
        0.35962678+0.14074839j, 0. +0.j])

    The Catalyst compiler also supports capturing imperative Python control flow in compiled programs. You can
    enable this feature via the ``autograph=True`` keyword argument.
    
    Note AutoGraph results in additional
    restrictions, in particular whenever global state is involved. Please refer to the `AutoGraph guide <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/autograph.html>`__
    for a complete discussion of the supported and unsupported use-cases.

    .. code-block:: python

        @qjit(autograph=True)
        @qml.qnode(dev)
        def circuit(x: int):

            if x < 5:
                qml.Hadamard(wires=0)
            else:
                qml.T(wires=0)

            return qml.expval(qml.PauliZ(0))

    >>> circuit(3)
    array(0.)
    >>> circuit(5)
    array(1.)
    
For more details on using the :func:`~.qjit` decorator and Catalyst
with PennyLane, please refer to the Catalyst
`quickstart guide <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/quick_start.html>`__,
as well as the `sharp bits and debugging tips <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/sharp_bits.html>`__
page for an overview of the differences between Catalyst and PennyLane, and
how to best structure your workflows to improve performance when
using Catalyst.
"""

from .compiler import available_compilers, available, active

from .qjit_api import qjit
