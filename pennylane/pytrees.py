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
An internal module for working with pytrees.
"""
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple, Union

has_jax = True
try:
    import jax.tree_util as jax_tree_util
except ImportError:
    has_jax = False

Leaves = Any
Metadata = Any

FlattenFn = Callable[[Any], Tuple[Leaves, Metadata]]
UnflattenFn = Callable[[Leaves, Metadata], Any]


def flatten_list(obj: list):
    """Flatten a list."""
    return obj, None


def flatten_tuple(obj: tuple):
    """Flatten a tuple."""
    return obj, None


def flatten_dict(obj: dict):
    """Flatten a dictionary."""
    return obj.values(), tuple(obj.keys())


flatten_registrations: Dict[type, FlattenFn] = {
    list: flatten_list,
    tuple: flatten_tuple,
    dict: flatten_dict,
}


def unflatten_list(data, _) -> list:
    """Unflatten a list."""
    return data if isinstance(data, list) else list(data)


def unflatten_tuple(data, _) -> tuple:
    """Unflatten a tuple."""
    return tuple(data)


def unflatten_dict(data, metadata) -> dict:
    """Unflatten a dictinoary."""
    return dict(zip(metadata, data))


unflatten_registrations: Dict[type, UnflattenFn] = {
    list: unflatten_list,
    tuple: unflatten_tuple,
    dict: unflatten_dict,
}


def _register_pytree_with_pennylane(
    pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn
):
    flatten_registrations[pytree_type] = flatten_fn
    unflatten_registrations[pytree_type] = unflatten_fn


def _register_pytree_with_jax(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    def jax_unflatten(aux, parameters):
        return unflatten_fn(parameters, aux)

    jax_tree_util.register_pytree_node(pytree_type, flatten_fn, jax_unflatten)


def register_pytree(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    """Register a type with all available pytree backends.

    Current backends is jax.
    Args:
        pytree_type (type): the type to register, such as ``qml.RX``
        flatten_fn (Callable): a function that splits an object into trainable leaves and hashable metadata.
        unflatten_fn (Callable): a function that reconstructs an object from its leaves and metadata.

    Returns:
        None

    Side Effects:
        ``pytree`` type becomes registered with available backends.

    """

    _register_pytree_with_pennylane(pytree_type, flatten_fn, unflatten_fn)

    if has_jax:
        _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)


class Structure(namedtuple("Structure", ["type", "metadata", "children"])):
    """A pytree data structure, holding the type, metadata, and child pytree structures."""

    def __repr__(self):
        return f"PyTree({self.type.__name__}, {self.metadata}, {self.children})"


class Leaf:
    """A terminal node in a pytree."""

    def __repr__(self):
        return "Leaf"

    def __eq__(self, other):
        return isinstance(other, Leaf)

    def __hash__(self):
        return hash(Leaf)


leaf = Leaf()


def flatten(obj) -> Tuple[List[Any], Union[Structure, Leaf]]:
    """Flattens a pytree into leaves and a structure.

    Args:
        obj (Any): any object

    Returns:
        List[Any], Union[Structure, Leaf]: a list of leaves and a structure representing the object

    >>> op = qml.adjoint(qml.Rot(1.2, 2.3, 3.4, wires=0))
    >>> data, structure = flatten(op)
    >>> data
    [1.2, 2.3, 3.4]
    >>> structure
    <Tree(AdjointOperation, (), (<Tree(Rot, (<Wires = [0]>, ()), (Leaf, Leaf, Leaf))>,))>

    See also :function:`~.unflatten`.

    """
    flatten_fn = flatten_registrations.get(type(obj), None)
    if flatten_fn is None:
        return [obj], leaf
    leaves, metadata = flatten_fn(obj)

    flattened_leaves = []
    child_structures = []
    for l in leaves:
        child_leaves, child_structure = flatten(l)
        flattened_leaves += child_leaves
        child_structures.append(child_structure)

    structure = Structure(type(obj), metadata, child_structures)
    return flattened_leaves, structure


def unflatten(data: List[Any], structure: Union[Structure, Leaf]) -> Any:
    """Bind data to a structure to reconstruct a pytree object.

    Args:
        data (Iterable): iterable of numbers and numeric arrays
        structure (Structure, Leaf): The pytree structure object

    Returns:
        A repacked pytree.

    .. see-also:: :function:`~.flatten`

    >>> op = qml.adjoint(qml.Rot(1.2, 2.3, 3.4, wires=0))
    >>> data, structure = flatten(op)
    >>> unflatten([-2, -3, -4], structure)
    Adjoint(Rot(-2, -3, -4, wires=[0]))

    """
    return _unflatten(iter(data), structure)


def _unflatten(new_data, structure):
    if isinstance(structure, Leaf):
        return next(new_data)
    children = tuple(_unflatten(new_data, s) for s in structure[2])
    return unflatten_registrations[structure[0]](children, structure[1])
