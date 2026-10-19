import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch


src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.algorithms.lsh import LSHIndex
from models.record import VectorRecord


class TestLSHIndex(unittest.TestCase):
    """Kiểm tra LSH bằng dữ liệu nhỏ có thể tính đáp án bằng tay."""

    def build_with_fixed_planes(self, records):
        # Hai trục tạo 4 bucket theo dấu tọa độ: 11, 01, 10, 00.
        # Chỉ cố định pháp tuyến; vẫn chạy build/hash/search thật.
        index = LSHIndex(dimension=2, num_hyperplanes=2, seed=42)

        def generate_fixed_planes():
            index.hyperplanes = [[1.0, 0.0], [0.0, 1.0]]

        with patch.object(
            index, "_generate_hyperplanes", side_effect=generate_fixed_planes
        ):
            index.build(records)
        return index

    def test_generate_hyperplanes_shape_and_reproducibility(self):
        """Sinh đủ pháp tuyến đúng số chiều và tái tạo được với cùng seed."""
        index = LSHIndex(dimension=3, num_hyperplanes=4, seed=17)
        index._generate_hyperplanes()
        original = [plane[:] for plane in index.hyperplanes]

        self.assertEqual(len(original), 4)
        for plane in original:
            self.assertEqual(len(plane), 3)
            self.assertTrue(all(math.isfinite(value) for value in plane))
            self.assertGreater(sum(value * value for value in plane), 0)

        index._generate_hyperplanes()
        self.assertEqual(index.hyperplanes, original)
        other = LSHIndex(dimension=3, num_hyperplanes=4, seed=17)
        other._generate_hyperplanes()
        self.assertEqual(other.hyperplanes, original)

    def test_generate_does_not_change_global_random_state(self):
        """Sinh pháp tuyến không làm thay đổi bộ sinh ngẫu nhiên chung."""
        state = random.getstate()
        try:
            LSHIndex(dimension=2)._generate_hyperplanes()
            self.assertEqual(random.getstate(), state)
        finally:
            random.setstate(state)

    def test_hash_known_signs_boundary_and_positive_scaling(self):
        """Mã băm đúng theo dấu, xử lý tích bằng 0 và giữ nguyên khi nhân số dương."""
        index = LSHIndex(dimension=2, num_hyperplanes=3)
        index.hyperplanes = [[1, -1], [-1, 0], [0, 1]]

        self.assertEqual(index._hash_vector([3, 2]), "101")
        self.assertEqual(index._hash_vector([30, 20]), "101")
        self.assertEqual(index._hash_vector([-3, -2]), "010")
        # Tích vô hướng bằng 0 được quy ước thành bit 1.
        self.assertEqual(index._hash_vector([2, 2]), "101")

    def test_hash_rejects_wrong_dimension(self):
        """Từ chối băm vector có số chiều không khớp với chỉ mục."""
        index = self.build_with_fixed_planes([])
        for values in ([], [1], [1, 2, 3]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    index._hash_vector(values)

    def test_build_groups_records_into_known_buckets(self):
        """Các bản ghi được gom vào đúng bucket theo dấu tọa độ."""
        a = VectorRecord("A", "a", [3, 2])
        b = VectorRecord("B", "b", [6, 4])
        c = VectorRecord("C", "c", [-3, 2])
        d = VectorRecord("D", "d", [-3, -2])
        index = self.build_with_fixed_planes([a, b, c, d])

        self.assertEqual(index.buckets, {"11": [a, b], "01": [c], "00": [d]})

    def test_rebuild_replaces_old_records_without_duplicates(self):
        """Build lại thay dữ liệu cũ, không nhân đôi; build rỗng xóa chỉ mục."""
        a = VectorRecord("A", "a", [3, 2])
        b = VectorRecord("B", "b", [-3, -2])
        index = LSHIndex(dimension=2)
        index.build([a])
        index.build([b])
        index.build([b])

        stored = [record for bucket in index.buckets.values() for record in bucket]
        self.assertEqual(stored, [b])
        index.build([])
        self.assertEqual(index.buckets, {})
        self.assertEqual(index.search([1, 1], top_k=2), [])

    def test_invalid_build_preserves_previous_index(self):
        """Build gặp vector sai chiều báo lỗi và giữ nguyên chỉ mục trước đó."""
        a = VectorRecord("A", "a", [3, 2])
        index = self.build_with_fixed_planes([a])
        old_planes = [plane[:] for plane in index.hyperplanes]
        old_buckets = {key: bucket[:] for key, bucket in index.buckets.items()}
        invalid = VectorRecord("BAD", "bad", [1, 2, 3])

        with self.assertRaises(ValueError):
            index.build([a, invalid])

        self.assertEqual(index.hyperplanes, old_planes)
        self.assertEqual(index.buckets, old_buckets)

    def test_search_exact_bucket_ranks_by_cosine_distance(self):
        """Bucket trùng mã trả đúng top-k theo khoảng cách cosine tăng dần."""
        same = VectorRecord("SAME", "same", [2, 0])
        diagonal = VectorRecord("DIAG", "diagonal", [1, 1])
        perpendicular = VectorRecord("PERP", "perpendicular", [0, 3])
        opposite = VectorRecord("OPP", "opposite", [-1, 0])
        index = self.build_with_fixed_planes(
            [perpendicular, diagonal, opposite, same]
        )

        results = index.search([1, 0], top_k=3)
        self.assertEqual([record.id for record, _ in results], ["SAME", "DIAG", "PERP"])
        for (_, distance), expected in zip(results, [0, 1 - 1 / math.sqrt(2), 1]):
            self.assertAlmostEqual(distance, expected)
        self.assertEqual(index.search([1, 0], top_k=1)[0][0], same)

    def test_search_probes_all_one_bit_neighbors_and_keeps_best(self):
        """Xét hết bucket khác một bit để chọn ứng viên tốt nhất, bỏ bucket xa hai bit."""
        exact = VectorRecord("EXACT", "exact", [1, 1])
        first_neighbor = VectorRecord("FIRST", "first", [-10, 1])  # 01
        better_neighbor = VectorRecord("BETTER", "better", [1, -0.1])  # 10
        two_bits_away = VectorRecord("FAR", "far", [-1, -1])  # 00
        index = self.build_with_fixed_planes(
            [exact, first_neighbor, better_neighbor, two_bits_away]
        )

        # Không dừng sau bucket 01 dù lúc đó đã đủ 2 ứng viên.
        results = index.search([1, 1], top_k=2)
        self.assertEqual([record.id for record, _ in results], ["EXACT", "BETTER"])
        expanded = index.search([1, 1], top_k=10)
        self.assertEqual({record.id for record, _ in expanded}, {"EXACT", "FIRST", "BETTER"})
        self.assertEqual(len(expanded), 3)

    def test_search_finds_neighbor_when_exact_bucket_is_empty(self):
        """Bucket gốc rỗng vẫn tìm được bản ghi trong bucket lân cận một bit."""
        neighbor = VectorRecord("N", "neighbor", [-1, 2])
        far = VectorRecord("F", "far", [-1, -1])
        index = self.build_with_fixed_planes([neighbor, far])

        results = index.search([1, 1], top_k=2)
        self.assertEqual([record for record, _ in results], [neighbor])
        self.assertAlmostEqual(results[0][1], 1 - 1 / math.sqrt(10))

    def test_search_random_fallback_is_bounded_unique_and_repeatable(self):
        """Fallback lấy mẫu không trùng, đúng giới hạn, tái lập được và chấm điểm đúng."""
        # Tất cả bản ghi ở 00, cách mã truy vấn 11 đúng 2 bit.
        records = [
            VectorRecord(str(i), str(i), [-1, -i])
            for i in range(1, 6)
        ]
        index = self.build_with_fixed_planes(records)
        state = random.getstate()
        try:
            results = index.search([1, 1], top_k=2)
            self.assertEqual(results, index.search([1, 1], top_k=2))
            self.assertEqual(random.getstate(), state)
        finally:
            random.setstate(state)

        self.assertEqual(len(results), 2)
        self.assertEqual(len({record.id for record, _ in results}), 2)
        self.assertTrue(all(record in records for record, _ in results))
        distances = [distance for _, distance in results]
        self.assertEqual(distances, sorted(distances))
        for record, distance in results:
            i = int(record.id)
            self.assertAlmostEqual(distance, 1 + (1 + i) / math.sqrt(2 * (1 + i * i)))

        # K > N: fallback không được lấy mẫu quá số bản ghi có sẵn.
        all_results = index.search([1, 1], top_k=10)
        self.assertEqual({record.id for record, _ in all_results}, {record.id for record in records})

    def test_search_does_not_mutate_buckets_or_records(self):
        """Truy vấn nhiều lần không sửa bucket hay dữ liệu vector ban đầu."""
        records = [
            VectorRecord("A", "a", [1, 1]),
            VectorRecord("B", "b", [-1, 1]),
            VectorRecord("C", "c", [1, -1]),
        ]
        index = self.build_with_fixed_planes(records)
        original_buckets = {key: bucket[:] for key, bucket in index.buckets.items()}
        original_values = [record.values for record in records]

        first = index.search([1, 1], top_k=3)
        second = index.search([1, 1], top_k=3)
        self.assertEqual(first, second)
        self.assertEqual(index.buckets, original_buckets)
        self.assertEqual([record.values for record in records], original_values)

    def test_search_before_build_raises(self):
        """Search trước khi build phải báo RuntimeError."""
        with self.assertRaises(RuntimeError):
            LSHIndex(dimension=2).search([1, 1], top_k=1)

    def test_search_rejects_invalid_top_k(self):
        """Từ chối top_k không phải số nguyên dương, kể cả bool."""
        index = self.build_with_fixed_planes([])
        for top_k in (0, -1, True, False, 1.5, "2", None):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    index.search([1, 1], top_k=top_k)

    def test_search_rejects_invalid_queries(self):
        """Từ chối query sai chiều, vector không, NaN và vô cực."""
        index = self.build_with_fixed_planes([])
        for query in ([], [1], [1, 2, 3], [0, 0], [math.nan, 1], [math.inf, 1], [-math.inf, 1]):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    index.search(query, top_k=1)

    def test_zero_record_uses_project_cosine_convention(self):
        """Bản ghi vector không có khoảng cách cosine bằng 1 theo quy ước project."""
        zero = VectorRecord("ZERO", "zero", [0, 0])
        valid = VectorRecord("VALID", "valid", [1, 1])
        index = self.build_with_fixed_planes([zero, valid])
        results = index.search([1, 1], top_k=2)

        self.assertEqual([record.id for record, _ in results], ["VALID", "ZERO"])
        self.assertAlmostEqual(results[0][1], 0)
        self.assertAlmostEqual(results[1][1], 1)

    def test_end_to_end_with_generated_hyperplanes(self):
        """Build và search với pháp tuyến ngẫu nhiên tìm được các vector cùng hướng."""
        # Cùng hướng luôn cùng mã băm; không giả định LSH đạt recall 100%.
        a = VectorRecord("A", "a", [3, 2])
        b = VectorRecord("B", "b", [6, 4])
        c = VectorRecord("C", "c", [-3, -2])
        index = LSHIndex(dimension=2, num_hyperplanes=8, seed=42)
        index.build([c, b, a])

        results = index.search([3, 2], top_k=2)
        self.assertEqual({record.id for record, _ in results}, {"A", "B"})
        self.assertEqual(len(results), 2)
        for _, distance in results:
            self.assertAlmostEqual(distance, 0)

    def test_metric_validation(self):
        """Kiểm tra tính hợp lệ của tham số metric."""
        for valid_metric in ("cosine", "COSINE", "euclidean", "EUCLIDEAN_DISTANCE"):
            index = LSHIndex(dimension=2, metric=valid_metric)
            self.assertIn("cosine" if "cosine" in valid_metric.lower() else "euclidean", index.metric)

        with self.assertRaises(ValueError):
            LSHIndex(dimension=2, metric="manhattan")

    def test_search_euclidean_distance_computes_exact_l2(self):
        """Kiểm tra tìm kiếm LSH ở chế độ euclidean trả về khoảng cách Euclidean chính xác."""
        a = VectorRecord("A", "a", [3.0, 0.0])
        b = VectorRecord("B", "b", [0.0, 4.0])
        index = LSHIndex(dimension=2, num_hyperplanes=2, metric="euclidean", seed=42)
        index.build([a, b])

        results = index.search([3.0, 0.0], top_k=2)
        self.assertGreater(len(results), 0)
        top_rec, top_dist = results[0]
        self.assertEqual(top_rec.id, "A")
        self.assertAlmostEqual(top_dist, 0.0, places=5)

    def test_num_tables_validation_and_multi_table_search(self):
        """Kiểm tra xác thực num_tables và khả năng gom ứng viên từ nhiều bảng băm."""
        with self.assertRaises(ValueError):
            LSHIndex(dimension=2, num_tables=0)
        with self.assertRaises(ValueError):
            LSHIndex(dimension=2, num_tables=-2)

        # Multi-table index với L=3 bảng và metric euclidean
        index = LSHIndex(dimension=2, num_hyperplanes=4, num_tables=3, seed=42, metric="euclidean")
        records = [
            VectorRecord(f"R{i}", f"r{i}", [float(i), float(i * 2)])
            for i in range(1, 10)
        ]
        index.build(records)
        self.assertEqual(len(index.tables), 3)
        self.assertEqual(len(index.table_hyperplanes), 3)

        results = index.search([1.0, 2.0], top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0].id, "R1")
        self.assertAlmostEqual(results[0][1], 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
