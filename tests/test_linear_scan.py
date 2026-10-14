import math
from pathlib import Path
import sys
from typing import List, Tuple
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


def reference_linear_scan(
    records: List[VectorRecord],
    query: List[float],
    top_k: int,
    metric: str = "cosine",
) -> List[Tuple[VectorRecord, float]]:
    """
    Reference Implementation độc lập tuyệt đối (Oracle/Ground Truth):
    - Tự tính toán công thức thô theo định nghĩa toán học (không dùng metrics.py).
    - Không gọi qua LinearScanIndex.
    - Sắp xếp tường minh theo (distance, record.id).
    """
    if not records or top_k <= 0:
        return []

    candidates = []
    q_norm = math.sqrt(sum(q * q for q in query))
    is_cosine = metric not in ("euclidean", "euclidean_distance")
    if is_cosine and q_norm == 0.0:
        raise ValueError("Không thể tính Cosine Similarity với vector truy vấn có chuẩn độ dài bằng 0 (Zero Vector).")

    for rec in records:
        v = rec.values
        if not is_cosine:
            # Euclidean distance: sqrt(sum((q_i - v_i)^2))
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(query, v)))
        else:
            # Cosine distance: 1 - (dot(q, v) / (|q| * |v|))
            v_norm = math.sqrt(sum(x * x for x in v))
            if v_norm == 0.0:
                sim = 0.0
            else:
                dot = sum(x * y for x, y in zip(query, v))
                denom = q_norm * v_norm
                sim = dot / denom if denom > 0.0 else 0.0
                sim = max(-1.0, min(1.0, sim))
            dist = 1.0 - sim

        candidates.append((dist, rec.id, rec))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(item[2], item[0]) for item in candidates[:top_k]]


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

        query = [0.0, 0.0]
        results = index.search(query, top_k=4)

        self.assertEqual(len(results), 4)
        expected_ids = ["VEC_001", "VEC_002", "VEC_003", "VEC_004"]
        actual_ids = [rec.id for rec, _ in results]
        self.assertEqual(actual_ids, expected_ids)

        self.assertAlmostEqual(results[0][1], 0.0, places=5)
        self.assertAlmostEqual(results[1][1], math.sqrt(2.0), places=5)
        self.assertAlmostEqual(results[2][1], 3.0, places=5)
        self.assertAlmostEqual(results[3][1], 5.0, places=5)

    # 2. Dataset nhỏ có kết quả biết trước - Cosine Distance
    def test_cosine_known_results(self) -> None:
        records_cosine = [
            VectorRecord("VEC_A", "same_dir", [2.0, 0.0]),
            VectorRecord("VEC_B", "angle_45", [1.0, 1.0]),
            VectorRecord("VEC_C", "angle_90", [0.0, 3.0]),
            VectorRecord("VEC_D", "opposite", [-1.0, 0.0]),
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
        index.build(self.records_2d)

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
            VectorRecord("VEC_1", "w1", [1.0, 2.0]),
            VectorRecord("VEC_2", "w2", [1.0, 2.0, 3.0]),
        ]
        index = LinearScanIndex(metric="euclidean")
        with self.assertRaises(ValueError):
            index.build(invalid_records)

    def test_dimension_mismatch_in_search(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)

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
        tie_records = [
            VectorRecord("CHARLIE", "c", [0.0, 1.0]),
            VectorRecord("ALICE", "a", [1.0, 0.0]),
            VectorRecord("BOB", "b", [-1.0, 0.0]),
        ]
        index = LinearScanIndex(metric="euclidean")
        index.build(tie_records)

        results = index.search([0.0, 0.0], top_k=3)
        expected_ids = ["ALICE", "BOB", "CHARLIE"]
        actual_ids = [rec.id for rec, _ in results]
        self.assertEqual(actual_ids, expected_ids)

        results_again = index.search([0.0, 0.0], top_k=3)
        actual_ids_again = [rec.id for rec, _ in results_again]
        self.assertEqual(actual_ids_again, expected_ids)

    # 8. Kiểm thử các tham số đầu vào không hợp lệ
    def test_invalid_search_inputs(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)

        with self.assertRaises(ValueError):
            index.search([1.0, 1.0], top_k=0)
        with self.assertRaises(ValueError):
            index.search([1.0, 1.0], top_k=-2)
        with self.assertRaises(ValueError):
            index.search([], top_k=2)
        with self.assertRaises(ValueError):
            index.search([1.0, float("nan")], top_k=2)
        with self.assertRaises(ValueError):
            index.search([1.0, float("inf")], top_k=2)

    def test_invalid_top_k_types(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.records_2d)

        # Các kiểu dữ liệu và giá trị không hợp lệ của top_k
        invalid_cases = [0, -1, 2.5, True, False, "3", [1]]
        for invalid_k in invalid_cases:
            with self.assertRaises(ValueError):
                index.search([1.0, 1.0], top_k=invalid_k)  # type: ignore

    def test_zero_query_with_cosine_raises_value_error(self) -> None:
        index = LinearScanIndex(metric="cosine")
        index.build(self.records_2d)

        # Dataset không rỗng, query có norm = 0 đối với Cosine phải raise ValueError rõ ràng
        with self.assertRaises(ValueError):
            index.search([0.0, 0.0], top_k=2)

    def test_zero_vector_record_with_cosine(self) -> None:
        # Bản ghi có norm = 0 trong dataset
        records = [
            VectorRecord("ZERO_REC", "zero", [0.0, 0.0]),
            VectorRecord("VALID_REC", "valid", [1.0, 0.0]),
        ]
        index = LinearScanIndex(metric="cosine")
        index.build(records)

        # Query hợp lệ khác 0
        results = index.search([1.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        # VALID_REC trùng hướng -> dist = 0.0
        self.assertEqual(results[0][0].id, "VALID_REC")
        self.assertAlmostEqual(results[0][1], 0.0, places=5)
        # ZERO_REC góc không xác định, metrics.py quy ước sim = 0.0 -> dist = 1.0
        self.assertEqual(results[1][0].id, "ZERO_REC")
        self.assertAlmostEqual(results[1][1], 1.0, places=5)

    def test_empty_dataset_returns_empty_list_for_valid_queries(self) -> None:
        index = LinearScanIndex(metric="cosine")
        index.build([])

        # Dataset rỗng vẫn trả [] với input query hợp lệ
        self.assertEqual(index.search([1.0, 2.0], top_k=3), [])
        self.assertEqual(index.search([0.0, 0.0], top_k=3), [])

    def test_unsupported_metric(self) -> None:
        with self.assertRaises(ValueError):
            LinearScanIndex(metric="manhattan")


class TestBaselineCorrectnessVsReference(unittest.TestCase):
    """
    Kiểm chứng LinearScanIndex bằng Reference Implementation độc lập
    trên các tập dữ liệu nhỏ với đa dạng kịch bản.
    """

    def setUp(self) -> None:
        self.dataset_10 = [
            VectorRecord("ID_01", "w1", [1.2, 0.5, -0.3]),
            VectorRecord("ID_02", "w2", [-0.8, 1.5, 0.2]),
            VectorRecord("ID_03", "w3", [0.0, 0.0, 0.0]),  # Zero vector
            VectorRecord("ID_04", "w4", [2.0, 2.0, 2.0]),
            VectorRecord("ID_05", "w5", [-1.2, -0.5, 0.3]),
            VectorRecord("ID_06", "w6", [0.5, 0.5, 0.5]),
            VectorRecord("ID_07", "w7", [1.2, 0.5, -0.3]),  # Duplicate coords với ID_01
            VectorRecord("ID_08", "w8", [-2.0, -2.0, -2.0]),
            VectorRecord("ID_09", "w9", [0.1, -0.2, 0.4]),
            VectorRecord("ID_10", "w10", [1.0, -1.0, 1.0]),
        ]
        self.query = [0.8, 0.2, -0.1]

    def test_matches_reference_euclidean_all_k(self) -> None:
        index = LinearScanIndex(metric="euclidean")
        index.build(self.dataset_10)

        for k in (1, 3, 5, 10, 15):
            actual = index.search(self.query, top_k=k)
            expected = reference_linear_scan(self.dataset_10, self.query, top_k=k, metric="euclidean")

            self.assertEqual(len(actual), len(expected))
            for (act_rec, act_dist), (exp_rec, exp_dist) in zip(actual, expected):
                self.assertEqual(act_rec.id, exp_rec.id)
                self.assertAlmostEqual(act_dist, exp_dist, places=6)

    def test_matches_reference_cosine_all_k(self) -> None:
        index = LinearScanIndex(metric="cosine")
        index.build(self.dataset_10)

        for k in (1, 3, 5, 10, 15):
            actual = index.search(self.query, top_k=k)
            expected = reference_linear_scan(self.dataset_10, self.query, top_k=k, metric="cosine")

            self.assertEqual(len(actual), len(expected))
            for (act_rec, act_dist), (exp_rec, exp_dist) in zip(actual, expected):
                self.assertEqual(act_rec.id, exp_rec.id)
                self.assertAlmostEqual(act_dist, exp_dist, places=6)

    def test_matches_reference_zero_vector_query(self) -> None:
        index_cos = LinearScanIndex(metric="cosine")
        index_cos.build(self.dataset_10)

        zero_q = [0.0, 0.0, 0.0]
        # Query có norm = 0 đối với Cosine trên dataset không rỗng phải ném ValueError
        with self.assertRaises(ValueError):
            index_cos.search(zero_q, top_k=5)

    def test_matches_reference_duplicate_and_tie_breaking(self) -> None:
        # ID_01 và ID_07 có cùng tọa độ [1.2, 0.5, -0.3]
        index = LinearScanIndex(metric="euclidean")
        index.build(self.dataset_10)

        # Query đúng bằng tọa độ ID_01 / ID_07
        q_exact = [1.2, 0.5, -0.3]
        results = index.search(q_exact, top_k=2)

        # Hai kết quả đầu tiên phải là ID_01 và ID_07 với distance = 0.0
        self.assertEqual(results[0][0].id, "ID_01")
        self.assertEqual(results[1][0].id, "ID_07")
        self.assertAlmostEqual(results[0][1], 0.0, places=6)
        self.assertAlmostEqual(results[1][1], 0.0, places=6)

    def test_immutability_of_input_dataset(self) -> None:
        """Đảm bảo LinearScanIndex không làm thay đổi dữ liệu gốc trong records."""
        original_snapshots = [(r.id, r.label, list(r.values), r.norm) for r in self.dataset_10]

        index = LinearScanIndex(metric="cosine")
        index.build(self.dataset_10)

        # Thực thi nhiều truy vấn khác nhau
        index.search(self.query, top_k=5)
        index.search([0.5, 0.5, 0.5], top_k=3)
        index.search([1.0, 1.0, 1.0], top_k=10)

        # Kiểm tra dữ liệu gốc hoàn toàn bất biến
        current_snapshots = [(r.id, r.label, list(r.values), r.norm) for r in self.dataset_10]
        self.assertEqual(original_snapshots, current_snapshots)



class TestDatasetSmokeTest(unittest.TestCase):
    """
    Smoke Test trên tập con dữ liệu thực tế data/vectors.txt (tối đa 10.000 dòng).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.file_path = root_dir / "data" / "vectors.txt"
        cls.sample_records = []

        if cls.file_path.exists():
            # Đọc tối đa 10.000 records với cơ chế phân tách chuẩn GloVe (parts[-50:])
            dim = 50
            with open(cls.file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if idx > 10000:
                        break
                    parts = line.strip().split()
                    if len(parts) >= dim + 1:
                        label = " ".join(parts[:-dim])
                        values = [float(x) for x in parts[-dim:]]
                        cls.sample_records.append(
                            VectorRecord(record_id=f"VEC_{idx:05d}", label=label, values=values)
                        )

    def test_smoke_subset_format_and_dimension(self) -> None:
        if not self.sample_records:
            self.skipTest("File data/vectors.txt không tồn tại.")

        self.assertGreater(len(self.sample_records), 0)
        self.assertLessEqual(len(self.sample_records), 10000)
        self.assertEqual(self.sample_records[0].dimension, 50)

    def test_smoke_query_cosine_and_euclidean(self) -> None:
        if not self.sample_records:
            self.skipTest("File data/vectors.txt không tồn tại.")

        # 1. Cosine search
        index_cos = LinearScanIndex(metric="cosine")
        index_cos.build(self.sample_records)

        query_the = self.sample_records[0].values  # Từ "the"
        res1 = index_cos.search(query_the, top_k=5)
        res2 = index_cos.search(query_the, top_k=5)

        # Kết quả đúng định dạng
        self.assertEqual(len(res1), 5)
        self.assertIsInstance(res1[0][0], VectorRecord)
        self.assertIsInstance(res1[0][1], float)

        # Rank 1 chính là từ "the" với distance = 0.0
        self.assertEqual(res1[0][0].id, self.sample_records[0].id)
        self.assertAlmostEqual(res1[0][1], 0.0, places=5)

        # Sort tăng dần theo distance
        distances = [d for _, d in res1]
        self.assertEqual(distances, sorted(distances))

        # Không có duplicate IDs
        ids = [r.id for r, _ in res1]
        self.assertEqual(len(ids), len(set(ids)))

        # Determinism: cùng input cho cùng kết quả
        self.assertEqual([(r.id, d) for r, d in res1], [(r.id, d) for r, d in res2])

        # 2. Euclidean search
        index_euc = LinearScanIndex(metric="euclidean")
        index_euc.build(self.sample_records)
        res_euc = index_euc.search(query_the, top_k=5)

        self.assertEqual(len(res_euc), 5)
        self.assertEqual(res_euc[0][0].id, self.sample_records[0].id)
        self.assertAlmostEqual(res_euc[0][1], 0.0, places=5)
        distances_euc = [d for _, d in res_euc]
        self.assertEqual(distances_euc, sorted(distances_euc))


if __name__ == "__main__":
    unittest.main()
