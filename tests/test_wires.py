# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`pennylane.wires`.
"""
from importlib import import_module, util

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import WireError
from pennylane.wires import Wires

if util.find_spec("jax") is not None:
    jax = import_module("jax")
    jax_available = True
else:
    jax_available = False
    jax = None


# pylint: disable=too-many-public-methods, too-many-positional-arguments
class TestWires:
    """Tests for the ``Wires`` class."""

    def test_error_if_wires_none(self):
        """Tests that a TypeError is raised if None is given as wires."""
        with pytest.raises(TypeError, match="Must specify a set of wires."):
            Wires(None)

    @pytest.mark.parametrize("iterable", [np.array([0, 1, 2]), [0, 1, 2], (0, 1, 2), range(3)])
    def test_creation_from_common_iterables(self, iterable):
        """Tests that a Wires object can be created from standard iterable inputs."""

        wires = Wires(iterable)
        assert wires.labels == (0, 1, 2)

    @pytest.mark.parametrize(
        "iterable",
        [
            [qml.RX, qml.RY],
            [qml.PauliX],
            (None, qml.expval),
            (
                qml.device("default.qubit", wires=range(3)),
                qml.device("default.gaussian", wires=[qml.RX, 3]),
            ),
        ],
    )
    def test_creation_from_iterables_of_exotic_elements(self, iterable):
        """Tests that a Wires object can be created from standard iterable inputs."""

        wires = Wires(iterable)
        assert wires.labels == tuple(iterable)

    def test_creation_from_wires_object(self):
        """Tests that a Wires object can be created from another Wires object."""

        wires = Wires(Wires([0, 1, 2]))
        assert wires.labels == (0, 1, 2)

    def test_creation_from_wires_lists(self):
        """Tests that a Wires object can be created from a list of Wires."""

        wires = Wires([Wires([0]), Wires([1]), Wires([2])])
        assert wires.labels == (Wires([0]), Wires([1]), Wires([2]))

    @pytest.mark.parametrize(
        "iterable", [[1, 0, 4], ["a", "b", "c"], [0, 1, None], ["a", 1, "ancilla"]]
    )
    def test_creation_from_different_wire_types(self, iterable):
        """Tests that a Wires object can be created from iterables of different
        objects representing a single wire index."""

        wires = Wires(iterable)
        assert wires.labels == tuple(iterable)

    @pytest.mark.parametrize("wire", [1, -2, "a", "q1", -1.4])
    def test_creation_from_single_object(self, wire):
        """Tests that a Wires object can be created from a non-iterable object
        representing a single wire index."""

        wires = Wires(wire)
        assert wires.labels == (wire,)

    @pytest.mark.parametrize(
        "input", [[np.array([0, 1, 2]), np.array([3, 4])], [[0, 1, 2], [3, 4]], np.array(0.0)]
    )
    def test_error_for_incorrect_wire_types(self, input):
        """Tests that a Wires object cannot be created from unhashable objects such as np arrays or lists."""

        with pytest.raises(WireError, match="Wires must be hashable"):
            Wires(input)

    @pytest.mark.parametrize(
        "iterable", [np.array([4, 1, 1, 3]), [4, 1, 1, 3], (4, 1, 1, 3), ["a", "a", "b"]]
    )
    def test_error_for_repeated_wires(self, iterable):
        """Tests that a Wires object cannot be created from iterables with repeated indices."""

        with pytest.raises(WireError, match="Wires must be unique"):
            Wires(iterable)

    @pytest.mark.parametrize("iterable", [[4, 1, 0, 3], ["a", "b", "c"]])
    def test_indexing_and_slicing(self, iterable):
        """Tests the indexing and slicing of Wires objects."""

        wires = Wires(iterable)

        # check single index
        for wire, item in zip(wires, iterable):
            assert wire == item
        # check slicing
        assert wires[:2] == Wires(iterable[:2])

    def test_equality(self):
        """Tests that we can compare Wires objects with the '==' and '!=' operators."""

        wires1 = Wires([1, 2, 3])
        wires2 = Wires([3, 2, 1])
        wires3 = Wires([1, 2, 3])
        assert wires1 != wires2
        assert wires1 == wires3

    @pytest.mark.parametrize("iterable", [[4, 1, 0, 3], ["a", "b", "c"]])
    def test_length(self, iterable):
        """Tests that a Wires object returns the correct length."""

        wires = Wires(iterable)
        assert len(wires) == len(iterable)

    def test_contains(
        self,
    ):
        """Tests the __contains__() method."""

        wires = Wires([0, 1, 2, 3, Wires([4, 5]), None])

        assert 0 in wires
        assert Wires([4, 5]) in wires
        assert None in wires
        assert Wires([1]) not in wires
        assert Wires([0, 3]) not in wires
        assert Wires([0, 4]) not in wires

        assert [0, 4] not in wires
        assert [4] not in wires

    def test_contains_wires(
        self,
    ):
        """Tests the dedicated contains_wires() method."""

        wires = Wires([0, 1, 2, 3, Wires([4, 5]), None])

        assert wires.contains_wires(Wires([0, 3]))
        assert wires.contains_wires(Wires([1, 2, None]))
        assert wires.contains_wires(Wires([Wires([4, 5])]))  # Wires([4, 5]) is just a label!

        assert not wires.contains_wires(0)  # wrong type
        assert not wires.contains_wires([0, 1])  # wrong type
        assert not wires.contains_wires(
            Wires([4, 5])
        )  # looks up 4 and 5 in wires, which are not present

    def test_add_two_wires_objects(self):
        """Tests that wires objects add correctly."""
        wires1 = Wires([4, 0, 1])
        wires2 = Wires([1, 2])
        assert wires1 + wires2 == Wires([4, 0, 1, 2])

    def test_add_wires_object_with_iterable(self):
        """Tests that wires objects add correctly."""
        wires1 = [4, 0, 1]
        wires2 = Wires([1, 2])
        assert wires1 + wires2 == Wires([4, 0, 1, 2])
        assert wires2 + wires1 == Wires([1, 2, 4, 0])

    def test_add_wires_with_inbuilt_sum(self):
        """Tests that wires objects add correctly using sum()."""
        wires1 = [4, 0, 1]
        wires2 = Wires([1, 2])
        assert sum([wires1, wires2], Wires([]))

    def test_representation_and_string(self):
        """Tests the string representation via both __str__ and __repr__."""

        wires_str = str(Wires([1, 2, 3]))
        wires_repr = repr(Wires([1, 2, 3]))
        assert wires_str == "Wires([1, 2, 3])"
        assert wires_repr == "Wires([1, 2, 3])"

    def test_array_representation(self):
        """Tests that Wires object has an array representation."""

        wires = Wires([4, 0, 1])
        array = np.array(wires)
        assert isinstance(array, np.ndarray)
        assert array.shape == (3,)
        for w1, w2 in zip(array, np.array([4, 0, 1])):
            assert w1 == w2

    @pytest.mark.jax
    def test_jax_array_representation(self):
        """Tests that Wires object has a JAX array representation."""

        wires = Wires([4, 0, 1])
        array = jax.numpy.asarray(wires)
        assert isinstance(array, jax.numpy.ndarray)
        for w1, w2 in zip(array, [4, 0, 1]):
            assert w1 == w2

    def test_set_of_wires(self):
        """Tests that a set() of wires is formed correctly."""

        wires = Wires([0, 1, 2])
        list_of_wires = [Wires([1]), Wires([1]), Wires([1, 2, 3]), Wires([4])]

        assert set(wires) == {0, 1, 2}
        assert set(list_of_wires) == {Wires([1]), Wires([1, 2, 3]), Wires([4])}

    def test_label_property(self):
        """Tests the get_label() method."""

        labels = [0, "q1", 16]
        wires = Wires(labels)

        assert wires.labels == tuple(labels)
        assert wires.labels[1] == "q1"
        assert wires.labels[2] == 16

    def test_convert_to_numpy_array(self):
        """Tests that Wires object can be converted to a numpy array."""

        wires = Wires([4, 0, 1])
        array = wires.toarray()
        assert isinstance(array, np.ndarray)
        assert array.shape == (3,)
        for w1, w2 in zip(array, np.array([4, 0, 1])):
            assert w1 == w2

    def test_convert_to_list(self):
        """Tests that Wires object can be converted to a list."""

        wires = Wires([4, 0, 1])
        list_ = wires.tolist()
        assert isinstance(list_, list)
        assert list_ == [4, 0, 1]

    @pytest.mark.parametrize("iterable", [[4, 1, 0, 3], ["a", "b", "c"]])
    def test_index_method(self, iterable):
        """Tests the ``index()`` method."""

        wires = Wires(iterable)
        element = iterable[1]
        # check for non-Wires inputs
        assert wires.index(element) == 1
        # check for Wires inputs
        assert wires.index(Wires([element])) == 1
        # check that Wires of length >1 produce an error
        with pytest.raises(WireError, match="Can only retrieve index"):
            wires.index(Wires([1, 2]))
        # check that error raised when wire does not exist
        with pytest.raises(WireError, match="Wire with label d not found"):
            wires.index(Wires(["d"]))

    def test_indices_method(self):
        """Tests the ``indices()`` method."""

        wires = Wires([4, 0, 1])
        # for Wires inputs
        assert wires.indices(Wires([1, 4])) == [2, 0]
        # for non-Wires inputs
        assert wires.indices([1, 4]) == [2, 0]
        # for integer
        assert wires.indices(1) == [2]

    @pytest.mark.parametrize(
        "wires, wire_map, expected",
        [
            (Wires(["a", "b"]), {"a": 0, "b": 1}, Wires([0, 1])),
            (Wires([-1, 1]), {1: "c", -1: 1, "d": "e"}, Wires([1, "c"])),
        ],
    )
    def test_map_method(self, wires, wire_map, expected):
        """Tests the ``map()`` method."""

        assert wires.map(wire_map) == expected

        # error when labels not in wire_map dictionary
        with pytest.raises(WireError, match="No mapping for wire label"):
            wires.map({-1: Wires(4)})

        # error for non-unique wire labels
        with pytest.raises(WireError, match="Failed to implement wire map"):
            wires = Wires([0, 1])
            wires.map({0: "a", 1: "a"})

    def test_select_random_method(self):
        """Tests the ``select_random()`` method."""

        wires = Wires([4, 0, 1, 5, 6])

        assert len(wires.select_random(2)) == 2
        # check that seed makes call deterministic
        assert wires.select_random(4, seed=1) == wires.select_random(4, seed=1)

        with pytest.raises(WireError, match="Cannot sample"):
            wires.select_random(6)

    def test_subset_method(self):
        """Tests the ``subset()`` method."""

        wires = Wires([4, 0, 1, 5, 6])

        assert wires.subset([2, 3, 0]) == Wires([1, 5, 4])
        assert wires.subset(1) == Wires([0])
        assert wires.subset([4, 5, 7], periodic_boundary=True) == Wires([6, 4, 1])
        # if index does not exist
        with pytest.raises(WireError, match="Cannot subset wire at index"):
            wires.subset([10])

    def test_all_wires_method(self):
        """Tests the ``all_wires()`` method."""

        wires1 = Wires([2, 1, 3])
        wires2 = Wires([1, 4, 5, 2])
        wires3 = Wires([6, 5])

        new_wires = Wires.all_wires([wires1, wires2, wires3])
        assert new_wires.labels == (2, 1, 3, 4, 5, 6)

        new_wires = Wires.all_wires([wires1, wires2, wires3], sort=True)
        assert new_wires.labels == (1, 2, 3, 4, 5, 6)
        assert Wires.all_wires([[3, 4], [8, 5]]).labels == (3, 4, 8, 5)

    def test_shared_wires_method(self):
        """Tests the ``shared_wires()`` method."""

        wires1 = Wires([4, 0, 1])
        wires2 = Wires([3, 0, 4])
        wires3 = Wires([4, 0])
        res = Wires.shared_wires([wires1, wires2, wires3])
        assert res == Wires([4, 0])

        res = Wires.shared_wires([wires2, wires1, wires3])
        assert res == Wires([0, 4])

        with pytest.raises(WireError, match="Expected a Wires object"):
            Wires.shared_wires([[3, 4], [8, 5]])

    def test_unique_wires_method(self):
        """Tests the ``unique_wires()`` method."""

        wires1 = Wires([4, 0, 1])
        wires2 = Wires([3, 0, 4])
        wires3 = Wires([4, 0])
        res = Wires.unique_wires([wires1, wires2, wires3])
        assert res == Wires([1, 3])

        res = Wires.unique_wires([wires2, wires1, wires3])
        assert res == Wires([3, 1])

        with pytest.raises(WireError, match="Expected a Wires object"):
            Wires.unique_wires([[2, 1], [8, 5]])

    def test_equal_to_tuple(self):
        assert Wires([1, 2, 3]) == (1, 2, 3)
        assert Wires([1, 2, 3]) != (1, 5, 3)
        assert (1, 5, 3) != Wires([1, 2, 3])

    # pylint: disable=protected-access
    def test_hash_cached(self):
        """Test that the hash of a Wires object is being cached."""
        wires = Wires([0, 1])
        assert wires._hash is None
        h = hash(wires)
        assert wires._hash == h

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "source", [1, -2, "a", "q1", -1.4, np.array([0, 1, 2]), [0, 1, 2], (0, 1, 2), range(3)]
    )
    def test_wires_pytree(self, source):
        """Test that Wires class supports the PyTree flattening interface"""
        from jax.tree_util import tree_flatten, tree_unflatten

        wires = Wires(source)
        wires_flat, tree = tree_flatten(wires)
        wires2 = tree_unflatten(tree, wires_flat)
        assert isinstance(wires2, Wires), f"{wires2} is not Wires"
        assert wires == wires2, f"{wires} != {wires2}"

    @pytest.mark.parametrize(
        "wire_a, wire_b, expected",
        [
            (Wires([0, 1]), Wires([2, 3]), Wires([0, 1, 2, 3])),
            (Wires([0, 1]), [2, 3], Wires([0, 1, 2, 3])),
            ([], Wires([1, 2, 3]), Wires([1, 2, 3])),
            ({4, 5}, Wires([1, 2, 3]), Wires([1, 2, 3, 4, 5])),
            (Wires([1, 2]), Wires([1, 2, 3]), Wires([1, 2, 3])),
        ],
    )
    def test_union(self, wire_a, wire_b, expected):
        """
        Test the union operation (|) between two Wires objects.
        """
        assert wire_a | wire_b == expected
        assert wire_b | wire_a == expected

    @pytest.mark.parametrize(
        "wire_a, wire_b, expected",
        [
            (Wires([0, 1, 2]), Wires([2, 3, 4]), Wires([2])),
            (Wires([1, 2, 3]), Wires([]), Wires([])),
            (Wires([1, 2, 3]), [], Wires([])),
            (Wires([1, 2, 3]), "2", Wires([])),
            (Wires([1, 2, 3]), {3}, Wires([3])),
            (Wires([1, 2, 3]), Wires([1, 2, 3, 4]), Wires([1, 2, 3])),
        ],
    )
    def test_intersection(self, wire_a, wire_b, expected):
        """
        Test the intersection operation (&) between two Wires objects.
        """
        assert wire_a & wire_b == expected
        assert wire_b & wire_a == expected

    @pytest.mark.parametrize(
        "wire_a, wire_b, expected",
        [
            (Wires([0, 1, 2, 3]), Wires([2, 3, 4]), Wires([0, 1])),
            (Wires([1, 2, 3]), Wires([]), Wires([1, 2, 3])),
            (Wires([1, 2, 3]), Wires([1, 2, 3, 4]), Wires([])),
            (Wires([1, 2, 3]), [], Wires([1, 2, 3])),
            ([1, 2, 3], Wires([]), Wires([1, 2, 3])),
            ([], Wires([]), Wires([])),
            (Wires([]), [], Wires([])),
        ],
    )
    def test_difference(self, wire_a, wire_b, expected):
        """
        Test the difference operation (-) between two Wires objects.
        """
        assert wire_a - wire_b == expected

    @pytest.mark.parametrize(
        "wire_a, wire_b, expected",
        [
            (Wires([0, 1, 2]), Wires([2, 3, 4]), Wires([0, 1, 3, 4])),
            ([0, 1, 2], Wires([2, 3, 4]), Wires([0, 1, 3, 4])),
            (Wires([0, 1, 2]), [2, 3, 4], Wires([0, 1, 3, 4])),
            (Wires([]), Wires([1, 2, 3]), Wires([1, 2, 3])),
            (Wires([1, 2, 3]), Wires([1, 2, 3]), Wires([])),
        ],
    )
    def test_symmetric_difference(self, wire_a, wire_b, expected):
        """
        Test the symmetric difference operation (^) between two Wires objects.
        """
        assert wire_a ^ wire_b == expected

    @pytest.mark.parametrize(
        "wire_a, wire_b, wire_c, wire_d, expected",
        [
            (
                Wires([0, 1]),
                Wires([2, 3]),
                Wires([4, 5]),
                Wires([6, 7]),
                Wires([0, 1, 2, 3, 4, 5, 6, 7]),
            ),
            (Wires([0, 1]), Wires([1, 2]), Wires([2, 3]), Wires([3, 4]), Wires([0, 1, 2, 3, 4])),
            (Wires([]), Wires([1, 2]), Wires([2, 3]), Wires([3, 4, 5]), Wires([1, 2, 3, 4, 5])),
        ],
    )
    # pylint: disable=too-many-arguments
    def test_multiple_union(self, wire_a, wire_b, wire_c, wire_d, expected):
        """
        Test the union operation (|) with multiple Wires objects.
        """
        result = wire_a | wire_b | wire_c | wire_d
        assert result == expected
        assert wire_a.union(wire_b.union(wire_c.union(wire_d))) == expected

    def test_complex_operation(self):
        """
        Test a complex operation involving multiple set operations.
        This test combines union, intersection, difference, and symmetric difference operations.
        """
        wire_a = Wires([0, 1, 2, 3])
        wire_b = Wires([2, 3, 4, 5])
        wire_c = Wires([4, 5, 6, 7])
        wire_d = Wires([6, 7, 8, 9])

        # ((A ∪ B) ∩ (C ∪ D)) ^ ((A - D) ∪ (C - B))
        result = ((wire_a | wire_b) & (wire_c | wire_d)) ^ ((wire_a - wire_d) | (wire_c - wire_b))
        assert (wire_a | wire_b) & (wire_c | wire_d) == Wires([4, 5])
        assert (wire_a - wire_d) | (wire_c - wire_b) == Wires([0, 1, 2, 3, 6, 7])

        expected = Wires([0, 1, 2, 3, 4, 5, 6, 7])
        assert result == expected


@pytest.mark.jax
class TestWiresJax:
    """Tests the support for JAX arrays in the ``Wires`` class."""

    @pytest.mark.parametrize(
        "iterable, expected",
        (
            [
                (jax.numpy.array([0, 1, 2]), (0, 1, 2)),
                (jax.numpy.array([0]), (0,)),
                (jax.numpy.array(0), (0,)),
                (jax.numpy.array([]), ()),
            ]
            if jax_available
            else []
        ),
    )
    def test_creation_from_jax_array(self, iterable, expected):
        """Tests that a Wires object can be created from a JAX array."""
        wires = Wires(iterable)
        assert wires.labels == expected

    @pytest.mark.parametrize(
        "input",
        (
            [
                [jax.numpy.array([0, 1, 2]), jax.numpy.array([3, 4])],
                [jax.numpy.array([0, 1, 2]), 3],
                jax.numpy.array([[0, 1, 2]]),
                jax.numpy.array([[[0, 1], [2, 3]]]),
                jax.numpy.array([[[[0]]]]),
            ]
            if jax_available
            else []
        ),
    )
    def test_error_for_incorrect_jax_arrays(self, input):
        """Tests that a Wires object cannot be created from incorrect JAX arrays."""
        with pytest.raises(WireError, match="Wires must be hashable"):
            Wires(input)

    @pytest.mark.parametrize(
        "iterable",
        [jax.numpy.array([4, 1, 1, 3]), jax.numpy.array([0, 0])] if jax_available else [],
    )
    def test_error_for_repeated_wires_jax(self, iterable):
        """Tests that a Wires object cannot be created from a JAX array with repeated indices."""
        with pytest.raises(WireError, match="Wires must be unique"):
            Wires(iterable)

    def test_array_representation_jax(self):
        """Tests that Wires object has an array representation with JAX."""

        wires = Wires([4, 0, 1])
        array = jax.numpy.array(wires.labels)
        assert isinstance(array, jax.numpy.ndarray)
        assert array.shape == (3,)
        for w1, w2 in zip(array, jax.numpy.array([4, 0, 1])):
            assert w1 == w2

    @pytest.mark.parametrize(
        "source",
        (
            [jax.numpy.array([0, 1, 2]), jax.numpy.array([0]), jax.numpy.array(0)]
            if jax_available
            else []
        ),
    )
    def test_jax_wires_pytree(self, source):
        """Test that Wires class supports the PyTree flattening interface with JAX arrays."""

        wires = Wires(source)
        wires_flat, tree = jax.tree_util.tree_flatten(wires)
        wires2 = jax.tree_util.tree_unflatten(tree, wires_flat)
        assert isinstance(wires2, Wires), f"{wires2} is not Wires"
        assert wires == wires2, f"{wires} != {wires2}"
