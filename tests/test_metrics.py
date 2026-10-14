import math
from pathlib import Path
import sys
import unittest

# Đảm bảo đường dẫn import cho src và root project
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.metrics import cosine_similarity, dot_product, euclidean_distance



class TestSimilarityMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.v_a = [1.0, 2.0, 3.0]
        self.norm_a = math.sqrt(1.0**2 + 2.0**2 + 3.0**2)  # sqrt(14)
        self.v_b = [4.0, 5.0, 6.0]
        self.norm_b = math.sqrt(4.0**2 + 5.0**2 + 6.0**2)  # sqrt(77)

    # 1. Tests for dot_product
    def test_dot_product_correctness(self) -> None:
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        result = dot_product(self.v_a, self.v_b)
        self.assertAlmostEqual(result, 32.0, places=6)

    def test_dot_product_orthogonal(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(dot_product(v1, v2), 0.0, places=6)

    def test_dot_product_zero_vector(self) -> None:
        v1 = [0.0, 0.0, 0.0]
        self.assertAlmostEqual(dot_product(v1, self.v_a), 0.0, places=6)

    # 2. Tests for euclidean_distance
    def test_euclidean_distance_correctness(self) -> None:
        # sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9 + 9 + 9) = sqrt(27)
        expected = math.sqrt(27.0)
        self.assertAlmostEqual(euclidean_distance(self.v_a, self.v_b), expected, places=6)

    def test_euclidean_distance_identical(self) -> None:
        self.assertAlmostEqual(euclidean_distance(self.v_a, self.v_a), 0.0, places=6)

    def test_euclidean_distance_zero_vector(self) -> None:
        zero = [0.0, 0.0, 0.0]
        expected = math.sqrt(14.0)
        self.assertAlmostEqual(euclidean_distance(self.v_a, zero), expected, places=6)

    # 3. Tests for cosine_similarity
    def test_cosine_similarity_identical(self) -> None:
        sim = cosine_similarity(self.v_a, self.v_a, self.norm_a, self.norm_a)
        self.assertAlmostEqual(sim, 1.0, places=6)

    def test_cosine_similarity_opposite(self) -> None:
        v_opp = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(self.v_a, v_opp, self.norm_a, self.norm_a)
        self.assertAlmostEqual(sim, -1.0, places=6)

    def test_cosine_similarity_orthogonal(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = cosine_similarity(v1, v2, 1.0, 1.0)
        self.assertAlmostEqual(sim, 0.0, places=6)

    def test_cosine_similarity_zero_vector(self) -> None:
        zero = [0.0, 0.0, 0.0]
        sim = cosine_similarity(zero, self.v_a, 0.0, self.norm_a)
        self.assertEqual(sim, 0.0)

        sim_both = cosine_similarity(zero, zero, 0.0, 0.0)
        self.assertEqual(sim_both, 0.0)

    def test_cosine_similarity_clamping(self) -> None:
        # Test boundary clamping
        v = [1.0000000000000002]
        sim = cosine_similarity(v, v, 1.0, 1.0)
        self.assertLessEqual(sim, 1.0)
        self.assertGreaterEqual(sim, -1.0)

    # 4. Tests for edge cases and input validation
    def test_empty_vectors_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            dot_product([], [])
        with self.assertRaises(ValueError):
            euclidean_distance([], [])
        with self.assertRaises(ValueError):
            cosine_similarity([], [], 0.0, 0.0)

    def test_dimension_mismatch_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            dot_product([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            euclidean_distance([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0], 1.0, math.sqrt(5))

    def test_nan_values_raise_value_error(self) -> None:
        nan_vec = [1.0, float("nan")]
        valid_vec = [1.0, 2.0]
        with self.assertRaises(ValueError):
            dot_product(nan_vec, valid_vec)
        with self.assertRaises(ValueError):
            euclidean_distance(nan_vec, valid_vec)
        with self.assertRaises(ValueError):
            cosine_similarity(nan_vec, valid_vec, 1.0, 1.0)

    def test_inf_values_raise_value_error(self) -> None:
        inf_vec = [1.0, float("inf")]
        valid_vec = [1.0, 2.0]
        with self.assertRaises(ValueError):
            dot_product(inf_vec, valid_vec)
        with self.assertRaises(ValueError):
            euclidean_distance(inf_vec, valid_vec)
        with self.assertRaises(ValueError):
            cosine_similarity(inf_vec, valid_vec, 1.0, 1.0)

    def test_invalid_norm_in_cosine_similarity(self) -> None:
        # NaN / Inf norm
        with self.assertRaises(ValueError):
            cosine_similarity(self.v_a, self.v_b, float("nan"), self.norm_b)
        with self.assertRaises(ValueError):
            cosine_similarity(self.v_a, self.v_b, float("inf"), self.norm_b)
        # Negative norm
        with self.assertRaises(ValueError):
            cosine_similarity(self.v_a, self.v_b, -1.0, self.norm_b)


if __name__ == "__main__":
    unittest.main()
