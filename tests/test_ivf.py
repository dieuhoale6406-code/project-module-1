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

from core.algorithms.ivf import IVFIndex
from core.algorithms.linear_scan import LinearScanIndex
from core.storage import TextVectorStorage
from models.record import VectorRecord
import tempfile


class TestIVFIndex(unittest.TestCase):
    """Bộ kiểm thử toàn diện cho thuật toán IVF-Flat (IVFIndex)."""

    def setUp(self) -> None:
        # Tạo tập dữ liệu 3 chiều gồm 3 cụm rõ rệt:
        # Cụm A quanh (1, 1, 1), Cụm B quanh (10, 10, 10), Cụm C quanh (20, 20, 20)
        self.dim = 3
        self.sample_records = [
            VectorRecord("rec_01", "a1", [1.0, 1.0, 1.0]),
            VectorRecord("rec_02", "a2", [1.1, 0.9, 1.0]),
            VectorRecord("rec_03", "a3", [0.9, 1.1, 1.0]),
            VectorRecord("rec_04", "a4", [1.0, 1.2, 0.9]),
            VectorRecord("rec_05", "b1", [10.0, 10.0, 10.0]),
            VectorRecord("rec_06", "b2", [10.1, 9.9, 10.0]),
            VectorRecord("rec_07", "b3", [9.9, 10.1, 10.0]),
            VectorRecord("rec_08", "b4", [10.2, 10.0, 9.8]),
            VectorRecord("rec_09", "c1", [20.0, 20.0, 20.0]),
            VectorRecord("rec_10", "c2", [20.1, 19.9, 20.0]),
            VectorRecord("rec_11", "c3", [19.9, 20.1, 20.0]),
            VectorRecord("rec_12", "c4", [20.0, 20.2, 19.8]),
        ]

    # ==========================================
    # 1. Test Khởi tạo & Tham số hợp lệ (__init__)
    # ==========================================

    def test_init_valid_parameters(self) -> None:
        """Kiểm tra khởi tạo với tham số hợp lệ."""
        ivf = IVFIndex(dimension=50, num_clusters=8, nprobe=2, seed=42)
        self.assertEqual(ivf.dimension, 50)
        self.assertEqual(ivf.num_clusters, 8)
        self.assertEqual(ivf.nprobe, 2)
        self.assertEqual(ivf.seed, 42)
        self.assertEqual(ivf.name, "IVF-Flat")
        self.assertEqual(ivf.centroids, [])
        self.assertEqual(ivf.inverted_lists, {})

    def test_init_invalid_dimension(self) -> None:
        """Kiểm tra ném lỗi khi dimension <= 0."""
        with self.assertRaises(ValueError):
            IVFIndex(dimension=0)
        with self.assertRaises(ValueError):
            IVFIndex(dimension=-5)

    def test_init_invalid_num_clusters(self) -> None:
        """Kiểm tra ném lỗi khi num_clusters <= 0."""
        with self.assertRaises(ValueError):
            IVFIndex(dimension=3, num_clusters=0)
        with self.assertRaises(ValueError):
            IVFIndex(dimension=3, num_clusters=-1)

    def test_init_invalid_nprobe(self) -> None:
        """Kiểm tra ném lỗi khi nprobe <= 0."""
        with self.assertRaises(ValueError):
            IVFIndex(dimension=3, nprobe=0)
        with self.assertRaises(ValueError):
            IVFIndex(dimension=3, nprobe=-2)

    def test_init_metric_validation(self) -> None:
        """Kiểm tra chỉ chấp nhận metric Euclidean và từ chối các metric khác."""
        ivf1 = IVFIndex(dimension=3, metric="euclidean")
        self.assertEqual(ivf1.metric, "euclidean")

        ivf2 = IVFIndex(dimension=3, metric=" EUCLIDEAN_DISTANCE ")
        self.assertEqual(ivf2.metric, "euclidean_distance")

        with self.assertRaises(ValueError):
            IVFIndex(dimension=3, metric="cosine")
        with self.assertRaises(ValueError):
            IVFIndex(dimension=3, metric="manhattan")

    # ==========================================
    # 2. Test Huấn luyện K-Means (_train_kmeans)
    # ==========================================

    def test_train_kmeans_empty(self) -> None:
        """K-Means với danh sách bản ghi rỗng."""
        ivf = IVFIndex(dimension=3, num_clusters=3)
        ivf._train_kmeans([])
        self.assertEqual(ivf.centroids, [])

    def test_train_kmeans_fewer_records_than_clusters(self) -> None:
        """Khi số bản ghi N <= C: lấy toàn bộ bản ghi làm tâm."""
        small_records = self.sample_records[:2]
        ivf = IVFIndex(dimension=3, num_clusters=5)
        ivf._train_kmeans(small_records)
        self.assertEqual(len(ivf.centroids), 2)
        self.assertEqual(ivf.centroids[0], small_records[0].values)
        self.assertEqual(ivf.centroids[1], small_records[1].values)

    def test_train_kmeans_convergence(self) -> None:
        """K-Means phân cụm đúng 3 vùng dữ liệu độc lập."""
        ivf = IVFIndex(dimension=3, num_clusters=3, seed=42)
        ivf._train_kmeans(self.sample_records)
        self.assertEqual(len(ivf.centroids), 3)

        # Sắp xếp các centroid theo trục x để kiểm tra
        sorted_c = sorted(ivf.centroids, key=lambda c: c[0])
        # Cụm 1 xấp xỉ quanh 1.0, cụm 2 quanh 10.0, cụm 3 quanh 20.0
        self.assertAlmostEqual(sorted_c[0][0], 1.0, delta=0.5)
        self.assertAlmostEqual(sorted_c[1][0], 10.0, delta=0.5)
        self.assertAlmostEqual(sorted_c[2][0], 20.0, delta=0.5)

    def test_train_kmeans_reproducibility(self) -> None:
        """Cùng seed phải cho ra kết quả centroids giống hệt nhau."""
        ivf1 = IVFIndex(dimension=3, num_clusters=3, seed=123)
        ivf1._train_kmeans(self.sample_records)

        ivf2 = IVFIndex(dimension=3, num_clusters=3, seed=123)
        ivf2._train_kmeans(self.sample_records)

        self.assertEqual(ivf1.centroids, ivf2.centroids)

    # ==========================================
    # 3. Test Dựng Chỉ mục đảo (build)
    # ==========================================

    def test_build_empty(self) -> None:
        """Build với tập rỗng."""
        ivf = IVFIndex(dimension=3, num_clusters=3)
        ivf.build([])
        self.assertEqual(ivf.centroids, [])
        self.assertEqual(ivf.inverted_lists, {})

    def test_build_mismatched_dimension(self) -> None:
        """Bắt lỗi khi có vector không đúng dimension."""
        invalid_rec = VectorRecord("err", "err", [1.0, 2.0])  # 2D thay vì 3D
        ivf = IVFIndex(dimension=3, num_clusters=2)
        with self.assertRaises(ValueError):
            ivf.build(self.sample_records + [invalid_rec])

    def test_build_partitions_all_records(self) -> None:
        """Đảm bảo mọi bản ghi đều được xếp vào inverted_lists và bảo toàn số lượng."""
        ivf = IVFIndex(dimension=3, num_clusters=3, seed=42)
        ivf.build(self.sample_records)

        self.assertEqual(len(ivf.centroids), 3)
        self.assertEqual(set(ivf.inverted_lists.keys()), {0, 1, 2})

        total_in_lists = sum(len(cluster) for cluster in ivf.inverted_lists.values())
        self.assertEqual(total_in_lists, len(self.sample_records))

        # Mỗi cụm phải chứa đúng 4 phần tử tương ứng với 3 nhóm ban đầu
        for cluster_records in ivf.inverted_lists.values():
            self.assertEqual(len(cluster_records), 4)

    # ==========================================
    # 4. Test Tìm kiếm Top-K (search)
    # ==========================================

    def test_search_unbuilt_index(self) -> None:
        """Search trên index chưa build trả về []."""
        ivf = IVFIndex(dimension=3, num_clusters=2)
        results = ivf.search([1.0, 1.0, 1.0], top_k=3)
        self.assertEqual(results, [])

    def test_search_invalid_top_k(self) -> None:
        """Bắt lỗi top_k không hợp lệ."""
        ivf = IVFIndex(dimension=3, num_clusters=2)
        ivf.build(self.sample_records)
        with self.assertRaises(ValueError):
            ivf.search([1.0, 1.0, 1.0], top_k=0)
        with self.assertRaises(ValueError):
            ivf.search([1.0, 1.0, 1.0], top_k=-3)
        with self.assertRaises(ValueError):
            ivf.search([1.0, 1.0, 1.0], top_k=True)  # type: ignore

    def test_search_invalid_query_values(self) -> None:
        """Bắt lỗi khi query rỗng, sai số chiều hoặc chứa NaN/Inf."""
        ivf = IVFIndex(dimension=3, num_clusters=2)
        ivf.build(self.sample_records)

        with self.assertRaises(ValueError):
            ivf.search([], top_k=3)
        with self.assertRaises(ValueError):
            ivf.search([1.0, 2.0], top_k=3)  # Sai dimension
        with self.assertRaises(ValueError):
            ivf.search([1.0, float("nan"), 1.0], top_k=3)
        with self.assertRaises(ValueError):
            ivf.search([1.0, float("inf"), 1.0], top_k=3)

    def test_search_correctness_and_ordering(self) -> None:
        """Tìm kiếm trả về kết quả đúng số lượng và khoảng cách tăng dần."""
        ivf = IVFIndex(dimension=3, num_clusters=3, nprobe=2, seed=42)
        ivf.build(self.sample_records)

        query = [1.0, 1.0, 1.0]
        results = ivf.search(query, top_k=3)

        self.assertEqual(len(results), 3)

        # Kiểm tra thứ tự tăng dần theo khoảng cách
        distances = [dist for _, dist in results]
        self.assertEqual(distances, sorted(distances))

        # Điểm đầu tiên phải là rec_01 với khoảng cách 0.0
        best_rec, best_dist = results[0]
        self.assertEqual(best_rec.id, "rec_01")
        self.assertAlmostEqual(best_dist, 0.0, places=6)

    def test_search_recall_with_full_probe(self) -> None:
        """Khi nprobe = num_clusters, IVF duyệt toàn bộ cụm và phải đạt 100% Recall so với Linear Scan."""
        ivf = IVFIndex(dimension=3, num_clusters=3, nprobe=3, seed=42)
        ivf.build(self.sample_records)

        linear = LinearScanIndex(metric="euclidean")
        linear.build(self.sample_records)

        query = [9.95, 10.05, 10.0]
        k = 4

        ivf_results = ivf.search(query, top_k=k)
        linear_results = linear.search(query, top_k=k)

        ivf_ids = [rec.id for rec, _ in ivf_results]
        linear_ids = [rec.id for rec, _ in linear_results]

        self.assertEqual(ivf_ids, linear_ids)

        for (_, d_ivf), (_, d_lin) in zip(ivf_results, linear_results):
            self.assertAlmostEqual(d_ivf, d_lin, places=6)


class TestTextVectorStorage(unittest.TestCase):
    """Kiểm thử tính toàn vẹn dữ liệu cho tầng lưu trữ TextVectorStorage."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "vectors.txt"
        self.storage = TextVectorStorage(file_path=str(self.file_path), expected_dim=3)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_nonexistent_file(self) -> None:
        """Đọc file chưa tồn tại trả về []."""
        storage = TextVectorStorage(file_path="nonexistent_file.txt", expected_dim=3)
        self.assertEqual(storage.load_vectors(), [])

    def test_append_and_load_normal(self) -> None:
        """Ghi và đọc dữ liệu thông thường thành công."""
        self.storage.append_vector("apple", [1.0, 2.0, 3.0])
        self.storage.append_vector("banana", [4.0, 5.0, 6.0])

        loaded = self.storage.load_vectors()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0][0], "apple")
        self.assertEqual(loaded[0][1], [1.0, 2.0, 3.0])
        self.assertEqual(loaded[1][0], "banana")
        self.assertEqual(loaded[1][1], [4.0, 5.0, 6.0])

    def test_append_collapses_whitespace_and_rejects_hashes(self) -> None:
        """Nhãn có xuống dòng/tab phải được gộp khoảng trắng; nhãn bắt đầu bằng # phải bị từ chối."""
        self.storage.append_vector("deep\nlearning\tword", [1.0, 2.0, 3.0])
        loaded = self.storage.load_vectors()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0], "deep learning word")

        with self.assertRaises(ValueError):
            self.storage.append_vector("#comment_tag", [1.0, 2.0, 3.0])

    def test_append_handles_missing_trailing_newline(self) -> None:
        """File cũ không kết thúc bằng newline thì bản ghi mới vẫn bắt đầu trên dòng riêng."""
        # Tạo file không có newline ở cuối
        with self.file_path.open("w", encoding="utf-8") as f:
            f.write("first 1.000000 2.000000 3.000000")  # Không có \n

        self.storage.append_vector("second", [4.0, 5.0, 6.0])

        loaded = self.storage.load_vectors()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0][0], "first")
        self.assertEqual(loaded[0][1], [1.0, 2.0, 3.0])
        self.assertEqual(loaded[1][0], "second")
        self.assertEqual(loaded[1][1], [4.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
