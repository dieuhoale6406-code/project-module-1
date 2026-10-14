import math
from pathlib import Path
import random
import sys
from typing import Any, List, Tuple
import unittest

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.structures.heap import BoundedMaxHeap
from models.record import VectorRecord


def reference_full_sorting(items: List[Tuple[float, Any]], top_k: int) -> List[Tuple[Any, float]]:
    """Reference full sorting implementation for Top-K verification."""
    if top_k <= 0 or not items:
        return []

    def sort_key(entry: Tuple[float, Any]) -> Tuple[float, str]:
        dist, item = entry
        rec_id = str(getattr(item, "id", str(item)))
        return (dist, rec_id)

    sorted_items = sorted(items, key=sort_key)
    top_candidates = sorted_items[:top_k]
    return [(item, dist) for dist, item in top_candidates]


class TestBoundedMaxHeapValidation(unittest.TestCase):
    """Validation tests for BoundedMaxHeap initialization and inputs."""

    def test_valid_capacity(self) -> None:
        heap = BoundedMaxHeap(capacity=5)
        self.assertEqual(heap.capacity, 5)
        self.assertEqual(heap.size(), 0)
        self.assertEqual(len(heap), 0)
        self.assertTrue(heap.is_empty())
        self.assertFalse(heap.is_full())
        self.assertIsNone(heap.peek_max())
        self.assertIsNone(heap.max_distance)

    def test_invalid_capacity(self) -> None:
        invalid_capacities = [0, -1, -10, 2.5, "5", None, [], {}, True, False]
        for cap in invalid_capacities:
            with self.subTest(capacity=cap):
                with self.assertRaises(ValueError):
                    BoundedMaxHeap(capacity=cap)  # type: ignore

    def test_invalid_distance_rejected(self) -> None:
        heap = BoundedMaxHeap(capacity=3)
        invalid_distances = [
            float("nan"),
            float("inf"),
            float("-inf"),
            "1.5",
            None,
            [1.0],
            True,
            False,
        ]
        for dist in invalid_distances:
            with self.subTest(distance=dist):
                with self.assertRaises(ValueError):
                    heap.push(dist, "item", item_id="item")  # type: ignore


class TestBoundedMaxHeapOperations(unittest.TestCase):
    """Functional tests for BoundedMaxHeap operations."""

    def test_k_equals_1(self) -> None:
        heap = BoundedMaxHeap(capacity=1)
        heap.push(10.0, "far", item_id="far")
        self.assertEqual(heap.size(), 1)
        self.assertTrue(heap.is_full())
        self.assertEqual(heap.peek_max(), (10.0, "far"))

        heap.push(5.0, "closer", item_id="closer")
        self.assertEqual(heap.size(), 1)
        self.assertEqual(heap.peek_max(), (5.0, "closer"))

        heap.push(8.0, "worse", item_id="worse")
        self.assertEqual(heap.size(), 1)
        self.assertEqual(heap.peek_max(), (5.0, "closer"))

        results = heap.get_sorted_results()
        self.assertEqual(results, [("closer", 5.0)])

    def test_k_greater_than_n(self) -> None:
        heap = BoundedMaxHeap(capacity=10)
        items = [(3.0, "item3"), (1.0, "item1"), (2.0, "item2")]
        for dist, item in items:
            heap.push(dist, item, item_id=item)

        self.assertEqual(heap.size(), 3)
        self.assertFalse(heap.is_full())
        self.assertEqual(heap.peek_max(), (3.0, "item3"))
        self.assertEqual(heap.max_distance, 3.0)

        results = heap.get_sorted_results()
        self.assertEqual(results, [("item1", 1.0), ("item2", 2.0), ("item3", 3.0)])

    def test_replacement_when_full(self) -> None:
        heap = BoundedMaxHeap(capacity=3)
        heap.push(5.0, "A", item_id="A")
        heap.push(8.0, "B", item_id="B")
        heap.push(3.0, "C", item_id="C")

        self.assertTrue(heap.is_full())
        self.assertEqual(heap.peek_max(), (8.0, "B"))

        heap.push(4.0, "D", item_id="D")
        self.assertEqual(heap.size(), 3)
        self.assertEqual(heap.peek_max(), (5.0, "A"))

        heap.push(6.0, "E", item_id="E")
        self.assertEqual(heap.size(), 3)
        self.assertEqual(heap.peek_max(), (5.0, "A"))

        results = heap.get_sorted_results()
        expected = [("C", 3.0), ("D", 4.0), ("A", 5.0)]
        self.assertEqual(results, expected)

    def test_tie_breaking_order(self) -> None:
        heap = BoundedMaxHeap(capacity=2)
        heap.push(2.0, "BOB", item_id="BOB")
        heap.push(2.0, "CHARLIE", item_id="CHARLIE")

        self.assertEqual(heap.peek_max(), (2.0, "CHARLIE"))

        heap.push(2.0, "ALICE", item_id="ALICE")
        self.assertEqual(heap.size(), 2)
        self.assertEqual(heap.peek_max(), (2.0, "BOB"))

        heap.push(2.0, "DANIEL", item_id="DANIEL")
        self.assertEqual(heap.size(), 2)
        self.assertEqual(heap.peek_max(), (2.0, "BOB"))

        results = heap.get_sorted_results()
        self.assertEqual(results, [("ALICE", 2.0), ("BOB", 2.0)])

    def test_vector_record_integration_and_tie_breaking(self) -> None:
        r_alice = VectorRecord("ALICE", "word_a", [1.0, 0.0])
        r_bob = VectorRecord("BOB", "word_b", [0.0, 1.0])
        r_charlie = VectorRecord("CHARLIE", "word_c", [-1.0, 0.0])
        r_daniel = VectorRecord("DANIEL", "word_d", [0.0, -1.0])

        heap = BoundedMaxHeap(capacity=3)
        heap.push(1.5, r_charlie)
        heap.push(1.5, r_daniel)
        heap.push(1.5, r_bob)

        self.assertEqual(heap.peek_max()[1].id, "DANIEL")

        heap.push(1.5, r_alice)
        self.assertEqual(heap.size(), 3)
        self.assertEqual(heap.peek_max()[1].id, "CHARLIE")

        results = heap.get_sorted_results()
        ids = [rec.id for rec, _ in results]
        self.assertEqual(ids, ["ALICE", "BOB", "CHARLIE"])
        self.assertTrue(all(math.isclose(dist, 1.5) for _, dist in results))

    def test_duplicate_items_and_ids(self) -> None:
        heap = BoundedMaxHeap(capacity=3)
        heap.push(1.0, "dup", item_id="ID1")
        heap.push(1.0, "dup", item_id="ID1")
        heap.push(1.0, "dup", item_id="ID1")

        self.assertEqual(heap.size(), 3)
        results = heap.get_sorted_results()
        self.assertEqual(len(results), 3)

    def test_clear_and_reuse(self) -> None:
        heap = BoundedMaxHeap(capacity=2)
        heap.push(1.0, "A", item_id="A")
        heap.push(2.0, "B", item_id="B")
        self.assertTrue(heap.is_full())

        heap.clear()
        self.assertEqual(heap.size(), 0)
        self.assertTrue(heap.is_empty())
        self.assertIsNone(heap.peek_max())

        heap.push(0.5, "C", item_id="C")
        self.assertEqual(heap.size(), 1)
        self.assertEqual(heap.get_sorted_results(), [("C", 0.5)])

    def test_empty_heap_results(self) -> None:
        heap = BoundedMaxHeap(capacity=5)
        self.assertEqual(heap.get_sorted_results(), [])
        self.assertIsNone(heap.peek_max())
        self.assertIsNone(heap.max_distance)


class TestBoundedMaxHeapInvariantsAndOracle(unittest.TestCase):
    """Heap invariant and oracle verification tests."""

    def _verify_max_heap_invariant(self, heap: BoundedMaxHeap) -> None:
        data = heap._heap
        n = len(data)
        for i in range(n):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n:
                self.assertGreaterEqual(
                    (data[i][0], data[i][1]),
                    (data[left][0], data[left][1]),
                    f"Heap invariant failed at index {i} vs left {left}",
                )
            if right < n:
                self.assertGreaterEqual(
                    (data[i][0], data[i][1]),
                    (data[right][0], data[right][1]),
                    f"Heap invariant failed at index {i} vs right {right}",
                )

    def test_randomized_heap_invariants_and_oracle(self) -> None:
        rng = random.Random(42)
        for trial in range(30):
            capacity = rng.randint(1, 20)
            n_elements = rng.randint(0, 50)
            heap = BoundedMaxHeap(capacity=capacity)

            raw_items = []
            for j in range(n_elements):
                dist = round(rng.uniform(0.0, 10.0), 2)
                item_id = f"ID_{rng.randint(1, 15):02d}"
                raw_items.append((dist, item_id))
                heap.push(dist, item_id, item_id=item_id)
                self._verify_max_heap_invariant(heap)
                self.assertLessEqual(heap.size(), capacity)

            heap_results = heap.get_sorted_results()
            expected_results = reference_full_sorting(raw_items, top_k=capacity)

            self.assertEqual(len(heap_results), len(expected_results))
            for (h_item, h_dist), (exp_item, exp_dist) in zip(heap_results, expected_results):
                self.assertEqual(h_item, exp_item)
                self.assertAlmostEqual(h_dist, exp_dist, places=5)


if __name__ == "__main__":
    unittest.main()
