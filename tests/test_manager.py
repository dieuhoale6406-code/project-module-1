import math
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

# Đảm bảo đường dẫn import cho src và root project
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.algorithms.base import BaseVectorIndex
from core.algorithms.linear_scan import LinearScanIndex
from core.manager import AlgorithmNotReadyError, VectorEngineManager
from models.record import VectorRecord


class MockLinkedStack:
    """Mock ngăn xếp hoạt động đầy đủ để kiểm thử logic Undo của Manager."""

    def __init__(self) -> None:
        self._items = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        return self._items.pop() if self._items else None

    def is_empty(self) -> bool:
        return len(self._items) == 0


class TestVectorEngineManager(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = VectorEngineManager()

    # 1. Khởi tạo Manager
    def test_init_empty(self) -> None:
        self.assertEqual(self.manager.record_count, 0)
        self.assertEqual(self.manager.dimension, 0)
        self.assertEqual(self.manager.get_all_records(), [])
        self.assertTrue(self.manager.is_stale)

    def test_init_with_initial_records(self) -> None:
        recs = [
            VectorRecord("V1", "apple", [1.0, 2.0]),
            VectorRecord("V2", "banana", [3.0, 4.0]),
        ]
        mgr = VectorEngineManager(recs)
        self.assertEqual(mgr.record_count, 2)
        self.assertEqual(mgr.dimension, 2)
        self.assertIsNotNone(mgr.get_record("apple"))
        self.assertIsNotNone(mgr.get_record("banana"))

    # 2. YÊU CẦU 2: Không ghi dữ liệu initial_records vào lịch sử Undo
    def test_initial_records_not_recorded_in_undo(self) -> None:
        recs = [
            VectorRecord("V1", "apple", [1.0, 2.0]),
            VectorRecord("V2", "banana", [3.0, 4.0]),
        ]
        mock_stack = MockLinkedStack()
        # Inject mock_stack ngay tại thời điểm khởi tạo trước khi nạp initial_records
        mgr = VectorEngineManager(recs, undo_stack=mock_stack)

        # 1. Kiểm tra trực tiếp stack: hoàn toàn rỗng sau quá trình khởi tạo
        self.assertTrue(mock_stack.is_empty())
        self.assertEqual(len(mock_stack._items), 0)

        # 2. Undo trả về None và không xóa các bản ghi khởi tạo
        msg = mgr.undo()
        self.assertIsNone(msg)
        self.assertEqual(mgr.record_count, 2)

        # 3. Thao tác thêm mới tiếp theo vẫn được ghi nhận vào Undo Stack bình thường
        mgr.add_record("cherry", [5.0, 6.0])
        self.assertEqual(len(mock_stack._items), 1)

        # 4. Hoàn tác chỉ xóa bản ghi mới ('cherry'), giữ nguyên các bản ghi khởi tạo ban đầu
        undo_msg = mgr.undo()
        self.assertIsNotNone(undo_msg)
        self.assertIn("cherry", undo_msg)
        self.assertEqual(mgr.record_count, 2)
        self.assertIsNotNone(mgr.get_record("apple"))
        self.assertIsNotNone(mgr.get_record("banana"))


    # 3. YÊU CẦU 1: Cập nhật _dimension chính xác trong Undo ADD và Undo DELETE
    def test_undo_add_resets_dimension_when_empty(self) -> None:
        mgr = VectorEngineManager()
        mgr.undo_stack = MockLinkedStack()

        # Ban đầu rỗng -> dimension = 0
        self.assertEqual(mgr.dimension, 0)

        # Thêm 1 vector 3 chiều
        mgr.add_record("cat", [1.0, 2.0, 3.0])
        self.assertEqual(mgr.record_count, 1)
        self.assertEqual(mgr.dimension, 3)

        # Hoàn tác ADD -> vector bị xóa, manager rỗng -> dimension phải reset về 0
        msg = mgr.undo()
        self.assertIsNotNone(msg)
        self.assertEqual(mgr.record_count, 0)
        self.assertEqual(mgr.dimension, 0)

    def test_undo_delete_restores_dimension(self) -> None:
        mgr = VectorEngineManager()
        mgr.undo_stack = MockLinkedStack()

        # Thêm vector 3 chiều
        mgr.add_record("dog", [4.0, 5.0, 6.0])
        self.assertEqual(mgr.dimension, 3)

        # Xóa vector -> manager rỗng -> dimension = 0
        mgr.delete_record("dog")
        self.assertEqual(mgr.record_count, 0)
        self.assertEqual(mgr.dimension, 0)

        # Hoàn tác DELETE -> vector được phục hồi -> dimension phải khôi phục về 3
        msg = mgr.undo()
        self.assertIsNotNone(msg)
        self.assertEqual(mgr.record_count, 1)
        self.assertEqual(mgr.dimension, 3)
        self.assertIsNotNone(mgr.get_record("dog"))

    def test_undo_update_preserves_dimension(self) -> None:
        mgr = VectorEngineManager()
        mgr.undo_stack = MockLinkedStack()

        mgr.add_record("bird", [1.0, 1.0])
        mgr.update_record("bird", [9.0, 9.0])
        self.assertEqual(mgr.get_record("bird").values, [9.0, 9.0])

        msg = mgr.undo()
        self.assertIsNotNone(msg)
        self.assertEqual(mgr.get_record("bird").values, [1.0, 1.0])
        self.assertEqual(mgr.dimension, 2)

    # 4. YÊU CẦU 3: Không âm thầm nuốt exceptions trong search_kdtree và tích hợp
    def test_search_kdtree_raises_clear_error_when_not_implemented(self) -> None:
        mgr = VectorEngineManager()
        mgr.add_record("sample", [1.0, 2.0])

        # KDTreeIndex hiện tại đang có raise NotImplementedError trong src/core/algorithms/kdtree.py
        # search_kdtree phải ném NotImplementedError rõ ràng, KHÔNG âm thầm nuốt lỗi trả về ([], 0.0)
        with self.assertRaises(NotImplementedError):
            mgr.search_kdtree([1.0, 2.0], top_k=1)

    def test_search_kdtree_raises_value_error_when_unregistered(self) -> None:
        mgr = VectorEngineManager()
        # Xóa kdtree khỏi registry để test trường hợp chưa đăng ký
        mgr._algorithm_registry.pop("kdtree", None)
        with self.assertRaises(ValueError):
            mgr.search_kdtree([1.0, 2.0], top_k=1)

    # 5. YÊU CẦU 4: Tách thời gian build index khỏi search latency
    def test_search_linear_latency_separated_from_build_time(self) -> None:
        self.manager.add_record("w1", [1.0, 0.0])
        self.manager.add_record("w2", [2.0, 0.0])

        # Gọi build_linear_index tường minh
        build_time = self.manager.build_linear_index(metric="euclidean")
        self.assertGreaterEqual(build_time, 0.0)
        self.assertEqual(self.manager.last_build_time_ms, build_time)

        # search_linear chỉ đo độ trễ truy vấn thuần túy
        results, latency_ms = self.manager.search_linear([0.0, 0.0], top_k=2, metric="euclidean")
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(latency_ms, 0.0)

    # 6. Thao tác CRUD cơ bản
    def test_add_and_get_record(self) -> None:
        rec = self.manager.add_record(label="king", values=[0.5, 0.8], record_id="REC_01")
        self.assertEqual(rec.label, "king")
        self.assertEqual(rec.id, "REC_01")
        self.assertEqual(self.manager.record_count, 1)
        self.assertEqual(self.manager.dimension, 2)

        # Lấy bằng label
        found_by_label = self.manager.get_record("king")
        self.assertIsNotNone(found_by_label)
        self.assertEqual(found_by_label.id, "REC_01")

        # Lấy bằng ID
        found_by_id = self.manager.get_record("REC_01")
        self.assertIsNotNone(found_by_id)
        self.assertEqual(found_by_id.label, "king")

        # Lấy nhãn không phân biệt hoa thường
        self.assertIsNotNone(self.manager.get_record("KING"))

    def test_update_record(self) -> None:
        self.manager.add_record("car", [1.0, 1.0])
        success = self.manager.update_record("car", [2.0, 2.0])
        self.assertTrue(success)
        rec = self.manager.get_record("car")
        self.assertEqual(rec.values, [2.0, 2.0])

        # Update từ không tồn tại
        self.assertFalse(self.manager.update_record("nonexistent", [1.0, 1.0]))

    def test_delete_record(self) -> None:
        self.manager.add_record("cat", [1.0, 2.0])
        self.manager.add_record("dog", [3.0, 4.0])
        self.assertEqual(self.manager.record_count, 2)

        deleted = self.manager.delete_record("cat")
        self.assertTrue(deleted)
        self.assertEqual(self.manager.record_count, 1)
        self.assertIsNone(self.manager.get_record("cat"))

        # Xóa lại từ đã xóa
        deleted_again = self.manager.delete_record("cat")
        self.assertFalse(deleted_again)

    # 7. Kiểm tra các điều kiện lỗi đầu vào (Input Validation)
    def test_duplicate_label_raises_error(self) -> None:
        self.manager.add_record("hello", [1.0, 2.0])
        with self.assertRaises(ValueError):
            self.manager.add_record("hello", [2.0, 3.0])
        with self.assertRaises(ValueError):
            self.manager.add_record("HELLO", [2.0, 3.0])

    def test_duplicate_id_raises_error(self) -> None:
        self.manager.add_record("w1", [1.0, 2.0], record_id="ID_001")
        with self.assertRaises(ValueError):
            self.manager.add_record("w2", [2.0, 3.0], record_id="ID_001")

    def test_dimension_mismatch_raises_error(self) -> None:
        self.manager.add_record("first", [1.0, 2.0])  # 2D
        with self.assertRaises(ValueError):
            self.manager.add_record("second", [1.0, 2.0, 3.0])  # 3D
        with self.assertRaises(ValueError):
            self.manager.update_record("first", [1.0, 2.0, 3.0])

    def test_invalid_values_nan_inf(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.add_record("nan_word", [1.0, float("nan")])
        with self.assertRaises(ValueError):
            self.manager.add_record("inf_word", [1.0, float("inf")])
        with self.assertRaises(ValueError):
            self.manager.add_record("", [1.0, 2.0])
        with self.assertRaises(ValueError):
            self.manager.add_record("empty_vec", [])

    # 8. Search Consistency sau CRUD
    def test_search_after_crud_does_not_return_stale_data(self) -> None:
        self.manager.add_record("word1", [1.0, 0.0])
        self.manager.add_record("word2", [2.0, 0.0])

        res, _ = self.manager.search_linear([0.0, 0.0], top_k=2, metric="euclidean")
        self.assertEqual(len(res), 2)
        self.assertEqual(self.manager.rebuild_count, 1)

        # Search lại cùng query/metric -> không rebuild index
        self.manager.search_linear([0.0, 0.0], top_k=2, metric="euclidean")
        self.assertEqual(self.manager.rebuild_count, 1)

        # Thêm từ mới -> index phải đánh dấu stale và rebuild khi search
        self.manager.add_record("word_near", [0.1, 0.0])
        self.assertTrue(self.manager.is_stale)

        res_after_add, _ = self.manager.search_linear([0.0, 0.0], top_k=1, metric="euclidean")
        self.assertEqual(self.manager.rebuild_count, 2)
        self.assertEqual(res_after_add[0][0].label, "word_near")

        # Xóa word_near -> index phải được cập nhật, không trả lại dữ liệu cũ đã xóa
        self.manager.delete_record("word_near")
        self.assertTrue(self.manager.is_stale)

        res_after_del, _ = self.manager.search_linear([0.0, 0.0], top_k=2, metric="euclidean")
        self.assertEqual(self.manager.rebuild_count, 3)
        labels = [rec.label for rec, _ in res_after_del]
        self.assertNotIn("word_near", labels)

    def test_search_empty_dataset(self) -> None:
        results, elapsed_ms = self.manager.search_linear([1.0, 2.0], top_k=5)
        self.assertEqual(results, [])
        self.assertGreaterEqual(elapsed_ms, 0.0)

    # 9. Extensibility Registry
    def test_algorithm_registry(self) -> None:
        self.assertEqual(self.manager.get_algorithm("linear_scan"), LinearScanIndex)

        class DummyIndex(BaseVectorIndex):
            def build(self, records):
                pass

            def search(self, query_values, top_k):
                return []

        self.manager.register_algorithm("dummy", DummyIndex)
        self.assertEqual(self.manager.get_algorithm("dummy"), DummyIndex)


def make_fake_index(build_error=None, search_error=None, build_delay=0.0):
    """
    Tạo class BaseVectorIndex giả CHỈ DÙNG TRONG TEST (brute-force Euclidean).
    Mỗi lần gọi trả về class mới với danh sách `created` riêng để đếm số lần dựng index.
    """

    class FakeIndex(BaseVectorIndex):
        created = []

        def __init__(self, dimension: int, ef: int = 1) -> None:
            super().__init__(name="Fake")
            self.dimension = dimension
            self.ef = ef
            self.records = []
            self.build_calls = 0
            self.search_calls = 0
            FakeIndex.created.append(self)

        def build(self, records) -> None:
            if build_error is not None:
                raise build_error
            if build_delay:
                time.sleep(build_delay)
            self.build_calls += 1
            self.records = list(records)

        def search(self, query_values, top_k):
            if search_error is not None:
                raise search_error
            self.search_calls += 1
            scored = sorted(
                ((math.dist(query_values, r.values), r.id, r) for r in self.records),
                key=lambda item: (item[0], item[1]),
            )
            return [(r, d) for d, _, r in scored[:top_k]]

    return FakeIndex


class MetricAwareFakeIndex(BaseVectorIndex):
    """Fake có constructor KHÁC (chỉ nhận metric, không nhận dimension) để kiểm tra Manager thích ứng."""

    created = []

    def __init__(self, metric: str = "cosine") -> None:
        super().__init__(name="MetricFake")
        self.metric = metric
        self.records = []
        MetricAwareFakeIndex.created.append(self)

    def build(self, records) -> None:
        self.records = list(records)

    def search(self, query_values, top_k):
        return [(r, 0.0) for r in self.records[:top_k]]


class TestUnifiedAlgorithmApi(unittest.TestCase):
    """Kiểm thử điều phối đa thuật toán của VectorEngineManager bằng index giả."""

    def setUp(self) -> None:
        self.manager = VectorEngineManager()
        self.manager.undo_stack = MockLinkedStack()
        self.manager.add_record("a", [0.0, 0.0], record_id="A")
        self.manager.add_record("b", [1.0, 0.0], record_id="B")
        self.manager.add_record("c", [5.0, 5.0], record_id="C")
        self.fake = make_fake_index()
        self.manager.register_algorithm("fake", self.fake)

    # 1. Registry & chọn thuật toán
    def test_default_registry_exposes_five_algorithm_ids(self) -> None:
        status = self.manager.list_algorithms()
        self.assertEqual(list(status), ["linear_scan", "kdtree", "lsh", "ivf", "hnsw", "fake"])
        # Đăng ký thành công chưa chứng minh thuật toán hoạt động
        self.assertEqual(status["lsh"], "registered")
        self.assertEqual(self.manager.get_algorithm_status("nope"), "unregistered")

    def test_register_and_select_algorithm(self) -> None:
        self.assertIs(self.manager.get_algorithm("fake"), self.fake)
        results, _ = self.manager.search("fake", [0.0, 0.0], top_k=2)
        self.assertEqual([r.id for r, _ in results], ["A", "B"])
        self.assertEqual(self.manager.get_algorithm_status("fake"), "ready")

    def test_register_rejects_invalid_registration(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.register_algorithm("bad", object)
        with self.assertRaises(ValueError):
            self.manager.register_algorithm("  ", self.fake)
        with self.assertRaises(ValueError):
            self.manager.register_algorithm("bad", self.fake, supported_metrics=())

    def test_unknown_algorithm_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.search("nope", [0.0, 0.0], top_k=1)
        with self.assertRaises(ValueError):
            self.manager.build_index("nope")

    # 2. Unified build & search
    def test_unified_linear_scan_matches_legacy_api(self) -> None:
        unified, _ = self.manager.search("linear_scan", [0.0, 0.0], 3, metric="euclidean")
        legacy, _ = self.manager.search_linear([0.0, 0.0], 3, metric="euclidean")
        self.assertEqual([(r.id, d) for r, d in unified], [(r.id, d) for r, d in legacy])
        self.assertEqual([r.id for r, _ in unified], ["A", "B", "C"])

    def test_search_validates_arguments(self) -> None:
        for bad_k in (0, -1, 2.5, True, "3"):
            with self.subTest(top_k=bad_k):
                with self.assertRaises(ValueError):
                    self.manager.search("fake", [0.0, 0.0], top_k=bad_k)
        with self.assertRaises(ValueError):
            self.manager.search("fake", [], top_k=1)
        with self.assertRaises(ValueError):
            self.manager.search("fake", [0.0, float("nan")], top_k=1)
        with self.assertRaises(ValueError):
            self.manager.search("fake", [0.0, 0.0, 0.0], top_k=1)  # sai số chiều
        self.assertEqual(self.fake.created, [])  # lỗi đầu vào không kích hoạt build

    def test_empty_dataset_returns_empty_without_building(self) -> None:
        mgr = VectorEngineManager()
        fake = make_fake_index()
        mgr.register_algorithm("fake", fake)
        self.assertEqual(mgr.search("fake", [1.0, 2.0], top_k=3), ([], 0.0))
        self.assertEqual(fake.created, [])

    def test_constructor_params_and_reconfiguration(self) -> None:
        self.manager.search("fake", [0.0, 0.0], 1, ef=4)
        self.assertEqual(len(self.fake.created), 1)
        self.assertEqual(self.fake.created[0].ef, 4)
        self.assertEqual(self.fake.created[0].dimension, 2)  # dimension do Manager cấp

        self.manager.search("fake", [0.0, 0.0], 1)  # không params: dùng lại index đang cache
        self.assertEqual(len(self.fake.created), 1)

        self.manager.search("fake", [0.0, 0.0], 1, ef=8)  # params khác: build lại
        self.assertEqual(len(self.fake.created), 2)
        self.assertEqual(self.fake.created[1].ef, 8)

    def test_invalid_algorithm_params_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.search("fake", [0.0, 0.0], 1, bogus=1)  # constructor không có tham số này
        with self.assertRaises(ValueError):
            self.manager.search("fake", [0.0, 0.0], 1, dimension=3)  # tham số do Manager cấp
        self.assertEqual(self.fake.created, [])

    def test_constructor_signature_adaptation(self) -> None:
        MetricAwareFakeIndex.created = []
        self.manager.register_algorithm("metric_fake", MetricAwareFakeIndex)
        self.manager.search("metric_fake", [0.0, 0.0], 1, metric="euclidean")
        self.assertEqual(len(MetricAwareFakeIndex.created), 1)
        self.assertEqual(MetricAwareFakeIndex.created[0].metric, "euclidean")

    # 3. Cache & invalidate
    def test_index_is_cached_and_reused(self) -> None:
        self.manager.search("fake", [0.0, 0.0], 1)
        self.manager.search("fake", [1.0, 0.0], 2)
        self.assertEqual(len(self.fake.created), 1)
        self.assertEqual(self.fake.created[0].build_calls, 1)
        self.assertEqual(self.fake.created[0].search_calls, 2)
        self.assertEqual(self.manager.rebuild_count, 1)
        self.assertFalse(self.manager.is_stale)

    def test_crud_invalidates_cached_index(self) -> None:
        self.manager.search("fake", [0.0, 0.0], 1)
        mutations = (
            lambda: self.manager.add_record("d", [0.1, 0.0], record_id="D"),
            lambda: self.manager.update_record("d", [0.2, 0.0]),
            lambda: self.manager.delete_record("d"),
        )
        for step, mutate in enumerate(mutations, start=1):
            with self.subTest(step=step):
                mutate()
                self.assertTrue(self.manager.is_stale)
                self.manager.search("fake", [0.0, 0.0], 1)
                self.assertEqual(len(self.fake.created), step + 1)
                self.assertFalse(self.manager.is_stale)

    def test_search_reflects_crud_changes(self) -> None:
        self.manager.search("fake", [0.0, 0.0], 1)
        self.manager.add_record("near", [0.01, 0.0], record_id="NEAR")
        results, _ = self.manager.search("fake", [0.01, 0.0], 1)
        self.assertEqual(results[0][0].id, "NEAR")
        self.manager.delete_record("near")
        results, _ = self.manager.search("fake", [0.01, 0.0], 1)
        self.assertNotIn("NEAR", [r.id for r, _ in results])

    def test_successful_undo_invalidates_cached_index(self) -> None:
        self.manager.add_record("d", [0.1, 0.0], record_id="D")
        self.manager.search("fake", [0.0, 0.0], 1)
        self.assertFalse(self.manager.is_stale)

        self.assertIsNotNone(self.manager.undo())  # undo ADD 'd'
        self.assertTrue(self.manager.is_stale)
        results, _ = self.manager.search("fake", [0.1, 0.0], 4)
        self.assertNotIn("D", [r.id for r, _ in results])
        self.assertEqual(len(self.fake.created), 2)

    def test_rejected_undo_keeps_cached_index_valid(self) -> None:
        self.manager.delete_record("a")  # DELETE được ghi vào Undo
        self.manager.add_record("a", [9.0, 9.0], record_id="A2", track_undo=False)  # xung đột label
        self.manager.search("fake", [9.0, 9.0], 1)
        rebuilds = self.manager.rebuild_count

        self.assertIsNone(self.manager.undo())  # bị từ chối an toàn, dữ liệu không đổi

        self.assertFalse(self.manager.is_stale)
        self.manager.search("fake", [9.0, 9.0], 1)
        self.assertEqual(self.manager.rebuild_count, rebuilds)
        self.assertEqual(len(self.fake.created), 1)

    def test_switching_algorithm_keeps_other_valid_index(self) -> None:
        other = make_fake_index()
        self.manager.register_algorithm("other", other)

        self.manager.search("fake", [0.0, 0.0], 1)
        self.manager.search("other", [0.0, 0.0], 1)
        self.manager.search("fake", [0.0, 0.0], 1)  # quay lại: index cũ vẫn dùng được
        self.assertEqual(len(self.fake.created), 1)
        self.assertEqual(len(other.created), 1)

        self.manager.add_record("d", [0.1, 0.0], record_id="D")  # đổi dữ liệu: cả hai lỗi thời
        self.manager.search("fake", [0.0, 0.0], 1)
        self.manager.search("other", [0.0, 0.0], 1)
        self.assertEqual(len(self.fake.created), 2)
        self.assertEqual(len(other.created), 2)

    def test_each_metric_has_its_own_cached_index(self) -> None:
        self.manager.search("linear_scan", [1.0, 1.0], 1, metric="cosine")
        self.manager.search("linear_scan", [1.0, 1.0], 1, metric="euclidean")
        self.manager.search("linear_scan", [1.0, 1.0], 1, metric="cosine_distance")  # alias, dùng lại
        self.assertEqual(self.manager.rebuild_count, 2)

    def test_reregistering_algorithm_drops_its_cached_index(self) -> None:
        self.manager.search("fake", [0.0, 0.0], 1)
        replacement = make_fake_index()
        self.manager.register_algorithm("fake", replacement)
        self.assertTrue(self.manager.is_stale)
        self.manager.search("fake", [0.0, 0.0], 1)
        self.assertEqual(len(replacement.created), 1)

    # 4. Metric / cấu hình không hỗ trợ
    def test_unsupported_metric_is_rejected_clearly(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.search("linear_scan", [0.0, 0.0], 1, metric="manhattan")

        strict = make_fake_index()
        self.manager.register_algorithm("euclid_only", strict, supported_metrics=("euclidean",))
        with self.assertRaises(ValueError):
            self.manager.search("euclid_only", [0.0, 0.0], 1, metric="cosine")
        self.assertEqual(strict.created, [])
        self.manager.search("euclid_only", [0.0, 0.0], 1)  # metric mặc định = metric đầu tiên được hỗ trợ
        self.assertEqual(self.manager.get_supported_metrics("euclid_only"), ("euclidean",))

    def test_default_algorithms_declare_supported_metrics(self) -> None:
        self.assertEqual(self.manager.get_supported_metrics("linear_scan"), ("cosine", "euclidean"))
        with self.assertRaises(ValueError):
            self.manager.search("kdtree", [0.0, 0.0], 1, metric="cosine")  # bị chặn trước khi build

    # 5. Module chưa triển khai không làm hỏng Manager
    def test_unimplemented_build_raises_clear_error_and_manager_survives(self) -> None:
        wip = make_fake_index(build_error=NotImplementedError("chưa làm"))
        self.manager.register_algorithm("wip", wip)

        with self.assertRaises(AlgorithmNotReadyError) as ctx:
            self.manager.search("wip", [0.0, 0.0], 1)
        self.assertIsInstance(ctx.exception, NotImplementedError)
        self.assertIn("wip", str(ctx.exception))
        self.assertEqual(self.manager.get_algorithm_status("wip"), "not_implemented")
        self.assertEqual(self.manager.rebuild_count, 0)  # không có build "giả" nào được ghi nhận
        self.assertTrue(self.manager.is_stale)

        results, _ = self.manager.search_linear([0.0, 0.0], 1, metric="euclidean")  # thuật toán khác vẫn chạy
        self.assertEqual(results[0][0].id, "A")

    def test_unimplemented_search_drops_index_and_reports_not_ready(self) -> None:
        wip = make_fake_index(search_error=NotImplementedError("search chưa làm"))
        self.manager.register_algorithm("wip", wip)
        with self.assertRaises(AlgorithmNotReadyError):
            self.manager.search("wip", [0.0, 0.0], 1)
        self.assertEqual(self.manager.get_algorithm_status("wip"), "not_implemented")
        self.assertTrue(self.manager.is_stale)

    def test_unimportable_algorithm_modules_do_not_break_manager(self) -> None:
        with mock.patch("importlib.import_module", side_effect=ImportError("module chưa có")):
            mgr = VectorEngineManager()
        mgr.add_record("a", [1.0, 0.0], record_id="A")

        self.assertEqual(mgr.get_algorithm_status("kdtree"), "unavailable")
        self.assertIn("kdtree", mgr.list_algorithms())
        with self.assertRaises(AlgorithmNotReadyError):
            mgr.search("kdtree", [1.0, 0.0], 1)
        results, _ = mgr.search_linear([1.0, 0.0], 1)  # Linear Scan không bị ảnh hưởng
        self.assertEqual(results[0][0].id, "A")

    # 6. Build time tách biệt search latency
    def test_build_time_is_separated_from_search_latency(self) -> None:
        slow = make_fake_index(build_delay=0.05)
        self.manager.register_algorithm("slow", slow)

        _, latency_ms = self.manager.search("slow", [0.0, 0.0], 1)  # search tự build index
        build_ms = self.manager.get_build_time_ms("slow")
        self.assertGreaterEqual(build_ms, 40.0)
        self.assertLess(latency_ms, 25.0)  # latency không chứa thời gian build
        self.assertEqual(self.manager.last_build_time_ms, build_ms)

        _, latency_again = self.manager.search("slow", [0.0, 0.0], 1)  # dùng cache: không build thêm
        self.assertEqual(self.manager.last_build_time_ms, build_ms)
        self.assertEqual(self.manager.rebuild_count, 1)
        self.assertLess(latency_again, 25.0)

    def test_explicit_build_index_returns_build_time(self) -> None:
        build_ms = self.manager.build_index("fake")
        self.assertGreaterEqual(build_ms, 0.0)
        self.assertEqual(self.manager.get_build_time_ms("fake"), build_ms)
        self.manager.search("fake", [0.0, 0.0], 1)  # đã build tường minh: search không build lại
        self.assertEqual(self.manager.rebuild_count, 1)
        self.assertIsNone(self.manager.get_build_time_ms("linear_scan"))  # chưa build

    # 7. API Linear Scan cũ vẫn hoạt động
    def test_legacy_linear_scan_api_still_works(self) -> None:
        build_ms = self.manager.build_linear_index(metric="euclidean")
        self.assertIsInstance(build_ms, float)
        self.assertEqual(self.manager.last_build_time_ms, build_ms)
        self.assertFalse(self.manager.is_stale)

        results, latency_ms = self.manager.search_linear([0.0, 0.0], 2, metric="euclidean")
        self.assertEqual([r.id for r, _ in results], ["A", "B"])
        self.assertGreaterEqual(latency_ms, 0.0)
        self.assertEqual(self.manager.rebuild_count, 1)

    def test_legacy_search_kdtree_routes_through_registry(self) -> None:
        kd_fake = make_fake_index()
        self.manager.register_algorithm("kdtree", kd_fake, supported_metrics=("euclidean",))
        results, _ = self.manager.search_kdtree([0.0, 0.0], 1, metric="cosine")  # metric cũ bị bỏ qua
        self.assertEqual(results[0][0].id, "A")
        self.manager.search_kdtree([0.0, 0.0], 1)
        self.assertEqual(len(kd_fake.created), 1)  # nay được cache thay vì build lại mỗi lần gọi


if __name__ == "__main__":
    unittest.main()
