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

from core.algorithms.linear_scan import LinearScanIndex
from models.record import VectorRecord


class TestLinearScanIndex(unittest.TestCase):
    def setUp(self) -> None:
        # Tập dữ liệu mẫu 2 chiều
        self.records_2d = [
            VectorRecord("VEC_001", "origin", [0.0, 0.0]),
            VectorRecord("VEC_002", "diag_1", [1.0, 1.0]),
            VectorRecord("VEC_003", "right_3", [3.0, 0.0]),
            VectorRecord("VEC_004", "far_point", [3.0, 4.0]),
        ]

    # 1. Dataset nhỏ có kết quả biết trước - Euclidean
    def test_euclidean_known_results(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)

        # Query tại gốc tọa độ [0.0, 0.0]
        query = [0.0, 0.0]
        results = index.search(query, top_k=4)

        self.assertEqual(len(results), 4)
        # Thứ tự khoảng cách Euclidean từ [0, 0]:
        # VEC_001: 0.0
        # VEC_002: sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414214
        # VEC_003: 3.0
        # VEC_004: sqrt(3^2 + 4^2) = 5.0
        expected_ids = ["VEC_001", "VEC_002", "VEC_003", "VEC_004"]
        actual_ids = [rec.id for rec, _ in results]
        self.assertEqual(actual_ids, expected_ids)

        # Kiểm tra giá trị khoảng cách
        self.assertAlmostEqual(results[0][1], 0.0, places=5)
        self.assertAlmostEqual(results[1][1], math.sqrt(2.0), places=5)
        self.assertAlmostEqual(results[2][1], 3.0, places=5)
        self.assertAlmostEqual(results[3][1], 5.0, places=5)

    # 2. Dataset nhỏ có kết quả biết trước - Cosine Distance
    def test_cosine_known_results(self) -> None:
        records_cosine = [
            VectorRecord("VEC_A", "same_dir", [2.0, 0.0]),        # cos_sim = 1.0 -> dist = 0.0
            VectorRecord("VEC_B", "angle_45", [1.0, 1.0]),        # cos_sim = 1/sqrt(2) ≈ 0.7071 -> dist ≈ 0.2929
            VectorRecord("VEC_C", "angle_90", [0.0, 3.0]),        # cos_sim = 0.0 -> dist = 1.0
            VectorRecord("VEC_D", "opposite", [-1.0, 0.0]),       # cos_sim = -1.0 -> dist = 2.0
        ]
        index = LinearScanIndex(metric="cosine")
        index.build(records_cosine)

        query = [1.0, 0.0]
        results = index.search(query, top_k=4)

        expected_order = ["VEC_A", "VEC_B", "VEC_C", "VEC_D"]
        actual_order = [rec.id for rec, _ in results]
        self.assertEqual(actual_order, expected_order)

        self.assertAlmostEqual(results[0][1], 0.0, places=5)
        self.assertAlmostEqual(results[1][1], 1.0 - (1.0 / math.sqrt(2.0)), places=5)
        self.assertAlmostEqual(results[2][1], 1.0, places=5)
        self.assertAlmostEqual(results[3][1], 2.0, places=5)

    # 3. Kiểm thử K = 1 và K > N
    def test_k_equals_1(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)

        results = index.search([0.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, "VEC_001")

    def test_k_greater_than_n(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)  # N = 4

        # Yêu cầu top_k = 10 (lớn hơn N)
        results = index.search([0.0, 0.0], top_k=10)
        self.assertEqual(len(results), 4)

    # 4. Kiểm thử Dataset rỗng
    def test_empty_dataset_after_build(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build([])
        results = index.search([1.0, 2.0], top_k=5)
        self.assertEqual(results, [])

    def test_empty_dataset_before_build(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        results = index.search([1.0, 2.0], top_k=5)
        self.assertEqual(results, [])

    # 5. Kiểm thử sai số chiều (Dimension mismatch)
    def test_dimension_mismatch_in_build(self) -> None:
        invalid_records = [
            VectorRecord("VEC_1", "w1", [1.0, 2.0]),       # 2D
            VectorRecord("VEC_2", "w2", [1.0, 2.0, 3.0]),  # 3D
        ]
        index = LinearScanIndex(metric="euclidean")
        with self.assertRaises(ValueError):
            index.build(invalid_records)

    def test_dimension_mismatch_in_search(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)  # 2D

        # Query 3D
        with self.assertRaises(ValueError):
            index.search([1.0, 2.0, 3.0], top_k=2)

    # 6. Kiểm thử vector trùng nhau (Identical vectors)
    def test_identical_vectors(self) -> None:
        identical_records = [
            VectorRecord("VEC_ID1", "clone1", [2.5, 3.5]),
            VectorRecord("VEC_ID2", "clone2", [2.5, 3.5]),
        ]
        index = LinearScanIndex(metric="euclidean")
        index.build(identical_records)

        results = index.search([2.5, 3.5], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0][1], 0.0, places=6)
        self.assertAlmostEqual(results[1][1], 0.0, places=6)

    # 7. Kiểm thử Tie-breaking theo (distance, record.id)
    def test_tie_breaking_order(self) -> None:
        # Cả 3 vector đều có cùng khoảng cách tới query [0.0, 0.0] (khoảng cách = 1.0)
        tie_records = [
            VectorRecord("CHARLIE", "c", [0.0, 1.0]),
            VectorRecord("ALICE", "a", [1.0, 0.0]),
            VectorRecord("BOB", "b", [-1.0, 0.0]),
        ]
        index = LinearScanIndex(metric="euclidean")
        index.build(tie_records)

        results = index.search([0.0, 0.0], top_k=3)
        # Sắp xếp tăng dần theo (distance, record.id) -> ALICE < BOB < CHARLIE
        expected_ids = ["ALICE", "BOB", "CHARLIE"]
        actual_ids = [rec.id for rec, _ in results]
        self.assertEqual(actual_ids, expected_ids)

        # Chạy lại để đảm bảo kết quả ổn định tuyệt đối
        results_again = index.search([0.0, 0.0], top_k=3)
        actual_ids_again = [rec.id for rec, _ in results_again]
        self.assertEqual(actual_ids_again, expected_ids)

    # 8. Kiểm thử các tham số đầu vào không hợp lệ
    def test_invalid_search_inputs(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)

        # top_k <= 0
        with self.assertRaises(ValueError):
            index.search([1.0, 1.0], top_k=0)
        with self.assertRaises(ValueError):
            index.search([1.0, 1.0], top_k=-2)

        # query rỗng
        with self.assertRaises(ValueError):
            index.search([], top_k=2)

        # query chứa NaN / Inf
        with self.assertRaises(ValueError):
            index.search([1.0, float("nan")], top_k=2)
        with self.assertRaises(ValueError):
            index.search([1.0, float("inf")], top_k=2)

    def test_unsupported_metric(self) -> None:
        with self.assertRaises(ValueError):
            LinearScanIndex(metric="manhattan")


if __name__ == "__main__":
    unittest.main()
