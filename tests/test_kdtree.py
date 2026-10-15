import math
import random
import sys
import unittest
from pathlib import Path
from typing import List, Tuple

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.algorithms.kdtree import KDTreeIndex
from models.record import VectorRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(label: str, values: List[float], idx: int = 0) -> VectorRecord:
    return VectorRecord(f"VEC_{idx:05d}", label, values)


def brute_force_knn(
    query: List[float], records: List[VectorRecord], top_k: int
) -> List[Tuple[str, float]]:
    """Ground-truth Top-K using full Euclidean scan."""
    dists = [
        (math.sqrt(sum((a - b) ** 2 for a, b in zip(query, r.values))), r.id)
        for r in records
    ]
    dists.sort(key=lambda x: (x[0], x[1]))
    return [(rid, d) for d, rid in dists[:top_k]]


def make_random_records(
    n: int, dim: int, seed: int = 0
) -> List[VectorRecord]:
    rng = random.Random(seed)
    return [
        make_record(f"w{i}", [rng.uniform(-1, 1) for _ in range(dim)], i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestKDTreeValidation — build() & search() input validation
# ---------------------------------------------------------------------------

class TestKDTreeValidation(unittest.TestCase):
    """Test that build() and search() reject invalid inputs correctly."""

    DIM = 4

    def _kd(self) -> KDTreeIndex:
        return KDTreeIndex(dimension=self.DIM)

    # --- build() ---

    def test_build_invalid_dimension_zero(self) -> None:
        kd = KDTreeIndex(dimension=0)
        with self.assertRaises(ValueError):
            kd.build([])

    def test_build_invalid_dimension_negative(self) -> None:
        kd = KDTreeIndex(dimension=-3)
        with self.assertRaises(ValueError):
            kd.build([])

    def test_build_records_not_list(self) -> None:
        kd = self._kd()
        with self.assertRaises(TypeError):
            kd.build(tuple())  # type: ignore

    def test_build_record_wrong_dimension(self) -> None:
        kd = self._kd()
        bad = make_record("bad", [1.0, 2.0], 0)   # dim=2, expected 4
        with self.assertRaises(ValueError):
            kd.build([bad])

    def test_build_record_contains_nan(self) -> None:
        kd = self._kd()
        with self.assertRaises(ValueError):
            VectorRecord("x", "bad", [1.0, float("nan"), 0.0, 0.0])

    def test_build_record_contains_inf(self) -> None:
        kd = self._kd()
        with self.assertRaises(ValueError):
            VectorRecord("x", "bad", [float("inf"), 0.0, 0.0, 0.0])

    def test_build_record_missing_values_attr(self) -> None:
        class BadRecord:
            id = "bad"
        kd = self._kd()
        with self.assertRaises((ValueError, AttributeError)):
            kd.build([BadRecord()])  # type: ignore

    # --- search() before build ---

    def test_search_on_empty_tree(self) -> None:
        kd = self._kd()
        kd.build([])
        self.assertEqual(kd.search([0.0] * self.DIM, top_k=5), [])

    # --- search() input validation ---

    def test_search_top_k_zero(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(ValueError):
            kd.search([0.0] * self.DIM, top_k=0)

    def test_search_top_k_negative(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(ValueError):
            kd.search([0.0] * self.DIM, top_k=-1)

    def test_search_top_k_bool_raises(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(TypeError):
            kd.search([0.0] * self.DIM, top_k=True)  # type: ignore

    def test_search_wrong_query_dimension(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(ValueError):
            kd.search([0.0] * (self.DIM + 1), top_k=1)

    def test_search_query_not_list(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(TypeError):
            kd.search("bad", top_k=1)  # type: ignore

    def test_search_query_contains_nan(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(ValueError):
            kd.search([float("nan")] * self.DIM, top_k=1)

    def test_search_query_contains_bool(self) -> None:
        kd = self._kd()
        kd.build(make_random_records(10, self.DIM))
        with self.assertRaises(TypeError):
            kd.search([True, 0.0, 0.0, 0.0], top_k=1)  # type: ignore


# ---------------------------------------------------------------------------
# TestKDTreeCorrectnessBasic — small, deterministic cases
# ---------------------------------------------------------------------------

class TestKDTreeCorrectnessBasic(unittest.TestCase):
    """Deterministic small-scale correctness tests."""

    def test_single_record_returns_itself(self) -> None:
        kd = KDTreeIndex(dimension=2)
        r = make_record("origin", [0.0, 0.0])
        kd.build([r])
        results = kd.search([1.0, 1.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, r.id)

    def test_nearest_is_closest(self) -> None:
        """Query [0,0] should return the record at [0,0] first."""
        kd = KDTreeIndex(dimension=2)
        records = [
            make_record("origin", [0.0, 0.0], 0),
            make_record("far1",   [5.0, 5.0], 1),
            make_record("far2",   [-5.0, 5.0], 2),
        ]
        kd.build(records)
        results = kd.search([0.0, 0.0], top_k=1)
        self.assertEqual(results[0][0].label, "origin")
        self.assertAlmostEqual(results[0][1], 0.0)

    def test_results_sorted_ascending_by_distance(self) -> None:
        """Results must be in ascending order of Euclidean distance."""
        records = make_random_records(100, dim=5, seed=7)
        kd = KDTreeIndex(dimension=5)
        kd.build(records)
        query = [0.0] * 5
        results = kd.search(query, top_k=10)
        dists = [d for _, d in results]
        self.assertEqual(dists, sorted(dists))

    def test_distances_are_correct_euclidean(self) -> None:
        """Returned distances must match manually computed Euclidean distances."""
        records = make_random_records(50, dim=3, seed=13)
        kd = KDTreeIndex(dimension=3)
        kd.build(records)
        query = [0.1, -0.2, 0.3]
        results = kd.search(query, top_k=5)
        for record, dist in results:
            expected = math.sqrt(sum((a - b) ** 2 for a, b in zip(query, record.values)))
            self.assertAlmostEqual(dist, expected, places=10)

    def test_top_k_greater_than_n_returns_all(self) -> None:
        """top_k > N must return all N records."""
        records = make_random_records(7, dim=3, seed=1)
        kd = KDTreeIndex(dimension=3)
        kd.build(records)
        results = kd.search([0.0, 0.0, 0.0], top_k=100)
        self.assertEqual(len(results), 7)

    def test_top_k_1_returns_exact_nearest(self) -> None:
        records = make_random_records(200, dim=4, seed=99)
        kd = KDTreeIndex(dimension=4)
        kd.build(records)
        query = [0.0] * 4
        kd_result = kd.search(query, top_k=1)[0]
        bf_id, bf_dist = brute_force_knn(query, records, 1)[0]
        self.assertEqual(kd_result[0].id, bf_id)
        self.assertAlmostEqual(kd_result[1], bf_dist, places=10)

    def test_build_does_not_mutate_input_list(self) -> None:
        """build() must not reorder the original records list."""
        records = make_random_records(20, dim=3, seed=5)
        original_ids = [r.id for r in records]
        kd = KDTreeIndex(dimension=3)
        kd.build(records)
        self.assertEqual([r.id for r in records], original_ids)

    def test_rebuild_replaces_old_tree(self) -> None:
        """Calling build() twice should replace the tree entirely."""
        kd = KDTreeIndex(dimension=2)
        batch1 = [make_record("a", [10.0, 10.0], 0)]
        batch2 = [make_record("b", [0.0, 0.0], 1)]
        kd.build(batch1)
        kd.build(batch2)
        results = kd.search([0.0, 0.0], top_k=1)
        self.assertEqual(results[0][0].label, "b")

    def test_duplicate_coordinates_handled(self) -> None:
        """All records at the same location — every result has distance 0."""
        records = [make_record("dup", [1.0, 1.0], i) for i in range(10)]
        kd = KDTreeIndex(dimension=2)
        kd.build(records)
        results = kd.search([1.0, 1.0], top_k=5)
        self.assertEqual(len(results), 5)
        for _, dist in results:
            self.assertAlmostEqual(dist, 0.0)

    def test_query_exactly_on_a_record(self) -> None:
        """Querying the exact coordinate of a record should return dist=0."""
        records = make_random_records(30, dim=3, seed=55)
        kd = KDTreeIndex(dimension=3)
        kd.build(records)
        exact_query = records[10].values
        results = kd.search(exact_query, top_k=1)
        self.assertAlmostEqual(results[0][1], 0.0, places=12)
        self.assertEqual(results[0][0].id, records[10].id)


# ---------------------------------------------------------------------------
# TestKDTreeCorrectnessVsBruteForce — randomized oracle tests
# ---------------------------------------------------------------------------

class TestKDTreeCorrectnessVsBruteForce(unittest.TestCase):
    """Randomized oracle: KD-Tree results must match brute-force exactly."""

    def _run_trial(self, n: int, dim: int, top_k: int, seed: int) -> None:
        records = make_random_records(n, dim, seed)
        kd = KDTreeIndex(dimension=dim)
        kd.build(records)

        rng = random.Random(seed + 1)
        for _ in range(10):
            query = [rng.uniform(-1, 1) for _ in range(dim)]
            kd_results = kd.search(query, top_k=top_k)
            bf_results = brute_force_knn(query, records, top_k)

            kd_ids = [r.id for r, _ in kd_results]
            bf_ids = [rid for rid, _ in bf_results]
            self.assertEqual(
                kd_ids, bf_ids,
                msg=f"Mismatch at dim={dim}, n={n}, k={top_k}, seed={seed}"
            )

    def test_dim2_small(self) -> None:
        self._run_trial(n=50, dim=2, top_k=5, seed=10)

    def test_dim3_medium(self) -> None:
        self._run_trial(n=200, dim=3, top_k=10, seed=20)

    def test_dim5_medium(self) -> None:
        self._run_trial(n=300, dim=5, top_k=10, seed=30)

    def test_dim10_medium(self) -> None:
        self._run_trial(n=500, dim=10, top_k=10, seed=40)

    def test_top_k_equals_1(self) -> None:
        self._run_trial(n=200, dim=4, top_k=1, seed=50)

    def test_top_k_equals_n(self) -> None:
        """top_k = N: must return all records in exact distance order."""
        records = make_random_records(20, dim=3, seed=60)
        kd = KDTreeIndex(dimension=3)
        kd.build(records)
        query = [0.0] * 3
        kd_results = kd.search(query, top_k=20)
        bf_results = brute_force_knn(query, records, 20)
        kd_ids = [r.id for r, _ in kd_results]
        bf_ids = [rid for rid, _ in bf_results]
        self.assertEqual(kd_ids, bf_ids)


# ---------------------------------------------------------------------------
# TestKDTreeInternalStructure — black-box checks on tree properties
# ---------------------------------------------------------------------------

class TestKDTreeInternalStructure(unittest.TestCase):
    """Verify internal state of the tree after build()."""

    def test_size_after_build(self) -> None:
        records = make_random_records(42, dim=3, seed=0)
        kd = KDTreeIndex(dimension=3)
        kd.build(records)
        self.assertEqual(kd._size, 42)

    def test_root_is_set_after_build(self) -> None:
        kd = KDTreeIndex(dimension=2)
        kd.build(make_random_records(5, dim=2))
        self.assertIsNotNone(kd.root)

    def test_root_is_none_after_empty_build(self) -> None:
        kd = KDTreeIndex(dimension=2)
        kd.build([])
        self.assertIsNone(kd.root)
        self.assertEqual(kd._size, 0)

    def test_axis_cycles_with_depth(self) -> None:
        """Root axis = 0; children axis = 1; grandchildren axis = 0 (dim=2)."""
        kd = KDTreeIndex(dimension=2)
        records = make_random_records(7, dim=2, seed=3)
        kd.build(records)
        root = kd.root
        self.assertEqual(root.axis, 0)
        if root.left:
            self.assertEqual(root.left.axis, 1)
        if root.right:
            self.assertEqual(root.right.axis, 1)

    def test_all_records_reachable(self) -> None:
        """Every record inserted must appear exactly once in the tree."""
        n = 25
        records = make_random_records(n, dim=3, seed=8)
        kd = KDTreeIndex(dimension=3)
        kd.build(records)

        visited_ids = set()

        def traverse(node):
            if node is None:
                return
            visited_ids.add(node.record.id)
            traverse(node.left)
            traverse(node.right)

        traverse(kd.root)
        expected_ids = {r.id for r in records}
        self.assertEqual(visited_ids, expected_ids)

    def test_quickselect_does_not_duplicate_records(self) -> None:
        """After build, no record appears twice in the tree."""
        records = make_random_records(31, dim=4, seed=9)
        kd = KDTreeIndex(dimension=4)
        kd.build(records)

        ids = []

        def collect(node):
            if node is None:
                return
            ids.append(node.record.id)
            collect(node.left)
            collect(node.right)

        collect(kd.root)
        self.assertEqual(len(ids), len(set(ids)), "Duplicate record in tree")


# ---------------------------------------------------------------------------
# TestKDTreeVectorRecord — integration with real VectorRecord
# ---------------------------------------------------------------------------

class TestKDTreeVectorRecord(unittest.TestCase):
    """Integration tests using actual VectorRecord objects."""

    def test_from_line_integration(self) -> None:
        line = "king 0.50451 0.68607 -0.59517 -0.022801 0.60046"
        r = VectorRecord.from_line(0, line)
        kd = KDTreeIndex(dimension=5)
        kd.build([r])
        results = kd.search(r.values, top_k=1)
        self.assertEqual(results[0][0].label, "king")
        self.assertAlmostEqual(results[0][1], 0.0, places=10)

    def test_values_property_is_copy(self) -> None:
        """Mutating the returned values list should NOT affect the tree."""
        r = make_record("test", [1.0, 2.0, 3.0])
        kd = KDTreeIndex(dimension=3)
        kd.build([r])
        # Mutate values returned by property
        vals = kd.root.record.values
        vals[0] = 999.0
        # Tree result should be unchanged
        results = kd.search([1.0, 2.0, 3.0], top_k=1)
        self.assertAlmostEqual(results[0][1], 0.0, places=10)


# ---------------------------------------------------------------------------
# TestKDTreeHighDimWarning — behavior at high dimensionality (correctness only)
# ---------------------------------------------------------------------------

class TestKDTreeHighDim(unittest.TestCase):
    """
    KD-Tree correctness at higher dimensions.
    Note: at d >= 20 pruning degrades severely (curse of dimensionality),
    so these tests verify correctness only, not speed.
    """

    def _assert_exact_match(self, dim: int, n: int, top_k: int, seed: int) -> None:
        records = make_random_records(n, dim, seed)
        kd = KDTreeIndex(dimension=dim)
        kd.build(records)
        rng = random.Random(seed + 99)
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        kd_ids = [r.id for r, _ in kd.search(query, top_k)]
        bf_ids = [rid for rid, _ in brute_force_knn(query, records, top_k)]
        self.assertEqual(kd_ids, bf_ids, f"Mismatch at dim={dim}")

    def test_dim20_correctness(self) -> None:
        self._assert_exact_match(dim=20, n=300, top_k=5, seed=100)

    def test_dim50_correctness(self) -> None:
        """GloVe 50d: exact results guaranteed, but search may be slow."""
        self._assert_exact_match(dim=50, n=200, top_k=5, seed=200)

    def test_dim100_correctness(self) -> None:
        """GloVe 100d: exact results guaranteed, but pruning nearly useless."""
        self._assert_exact_match(dim=100, n=100, top_k=3, seed=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
