import math
from pathlib import Path
import sys
import unittest

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.action import ActionRecord
from models.record import VectorRecord
from core.manager import VectorEngineManager


class MockLinkedStackWithPeek:
    """Mock ngăn xếp đầy đủ hỗ trợ cả peek(), pop(), push()."""

    def __init__(self) -> None:
        self._items = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        return self._items.pop() if self._items else None

    def peek(self):
        return self._items[-1] if self._items else None

    def is_empty(self) -> bool:
        return len(self._items) == 0


class MockLinkedStackNoPeek:
    """Mock ngăn xếp KHÔNG có peek() để kiểm thử nhánh fallback (pop + push rollback)."""

    def __init__(self) -> None:
        self._items = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        return self._items.pop() if self._items else None

    def is_empty(self) -> bool:
        return len(self._items) == 0


class TestCoreHardeningRegression(unittest.TestCase):
    # 1. VectorRecord.norm for zero vector
    def test_zero_vector_norm_is_strictly_zero(self) -> None:
        rec = VectorRecord("V_ZERO", "zero", [0.0, 0.0, 0.0])
        self.assertEqual(rec.norm, 0.0)

    # 2. NaN / Infinity validation on creation
    def test_nan_rejected_on_record_creation(self) -> None:
        with self.assertRaises(ValueError):
            VectorRecord("V_NAN", "nan_word", [1.0, float("nan")])

    def test_infinity_rejected_on_record_creation(self) -> None:
        with self.assertRaises(ValueError):
            VectorRecord("V_INF", "inf_word", [1.0, float("inf")])
        with self.assertRaises(ValueError):
            VectorRecord("V_NEGINF", "neginf_word", [float("-inf"), 0.0])

    # 3. Defensive copy preventing cached norm desync
    def test_record_values_immutability_defensive_copy(self) -> None:
        rec = VectorRecord("V_IMMUTABLE", "immutable", [1.0, 2.0])
        initial_norm = math.sqrt(5.0)
        self.assertAlmostEqual(rec.norm, initial_norm)

        # Thử mutate thông qua append
        vals = rec.values
        vals.append(999.0)
        self.assertEqual(rec.values, [1.0, 2.0])
        self.assertEqual(rec.dimension, 2)
        self.assertAlmostEqual(rec.norm, initial_norm)

        # Thử mutate thông qua gán phần tử
        rec.values[0] = 500.0
        self.assertEqual(rec.values[0], 1.0)
        self.assertAlmostEqual(rec.norm, initial_norm)

    # 4. Cross-type BaseEntity equality
    def test_base_entity_cross_type_inequality(self) -> None:
        rec = VectorRecord("SAME_ID", "word", [1.0, 2.0])
        act = ActionRecord("SAME_ID", "ADD", "word")
        self.assertFalse(rec == act)
        self.assertFalse(act == rec)
        self.assertNotEqual(rec, act)

    # 5. DELETE -> ADD cùng label khác ID, không ghi Undo -> Undo bị từ chối an toàn (ngăn silent data loss)
    def test_undo_delete_refused_on_readded_label_prevents_silent_data_loss(self) -> None:
        for stack_cls in (MockLinkedStackWithPeek, MockLinkedStackNoPeek):
            with self.subTest(stack=stack_cls.__name__):
                mgr = VectorEngineManager()
                mgr.undo_stack = stack_cls()

                # Thêm 'apple' ID=V1
                mgr.add_record("apple", [1.0, 2.0], record_id="V1")
                # Xóa 'apple' (đẩy DELETE vào undo)
                mgr.delete_record("apple")
                # Thêm lại 'apple' với ID khác V2 (không ghi undo để mô phỏng dữ liệu mới xen kẽ)
                mgr.add_record("apple", [3.0, 4.0], record_id="V2", track_undo=False)

                self.assertEqual(mgr.record_count, 1)
                self.assertEqual(mgr.get_record("apple").id, "V2")

                # Hoàn tác thao tác DELETE 'apple' (V1) -> PHẢI BỊ TỪ CHỐI do xung đột nhãn
                stack_size_before = len(mgr.undo_stack._items)
                undo_msg = mgr.undo()
                self.assertIsNone(undo_msg)

                # DỮ LIỆU MỚI KHÔNG BỊ XÓA ÂM THẦM: 'apple' vẫn là V2 với tọa độ [3.0, 4.0]
                self.assertEqual(mgr.record_count, 1)
                self.assertIn("V2", mgr._records)
                self.assertNotIn("V1", mgr._records)
                self.assertEqual(mgr.get_record("apple").id, "V2")
                self.assertEqual(mgr.get_record("apple").values, [3.0, 4.0])

                # ACTION KHÔNG BỊ MẤT KHỎI STACK: Kích thước ngăn xếp được bảo toàn
                self.assertEqual(len(mgr.undo_stack._items), stack_size_before)

                # Khi người dùng chủ động xóa V2, thử lại undo() sẽ THÀNH CÔNG khôi phục V1
                mgr.delete_record("apple", track_undo=False)
                retry_undo_msg = mgr.undo()
                self.assertIsNotNone(retry_undo_msg)
                self.assertEqual(mgr.record_count, 1)
                self.assertEqual(mgr.get_record("apple").id, "V1")
                self.assertEqual(mgr.get_record("apple").values, [1.0, 2.0])

    # 6. DELETE -> ADD trùng ID khác label, không ghi Undo -> Undo bị từ chối an toàn
    def test_undo_delete_refused_on_readded_id_prevents_silent_data_loss(self) -> None:
        for stack_cls in (MockLinkedStackWithPeek, MockLinkedStackNoPeek):
            with self.subTest(stack=stack_cls.__name__):
                mgr = VectorEngineManager()
                mgr.undo_stack = stack_cls()

                # Thêm 'apple' ID=V1
                mgr.add_record("apple", [1.0, 2.0], record_id="V1")
                mgr.delete_record("apple")

                # Thêm 'banana' sử dụng lại ID=V1
                mgr.add_record("banana", [5.0, 6.0], record_id="V1", track_undo=False)

                # Undo DELETE 'apple' (V1) -> PHẢI BỊ TỪ CHỐI do xung đột ID
                stack_size_before = len(mgr.undo_stack._items)
                undo_msg = mgr.undo()
                self.assertIsNone(undo_msg)

                # 'banana' KHÔNG BỊ XÓA ÂM THẦM
                self.assertEqual(mgr.record_count, 1)
                self.assertIn("V1", mgr._records)
                self.assertEqual(mgr.get_record("banana").id, "V1")
                self.assertEqual(mgr._label_to_id.get("banana"), "V1")
                self.assertNotIn("apple", mgr._label_to_id)

                # Action vẫn còn trên stack
                self.assertEqual(len(mgr.undo_stack._items), stack_size_before)

    # 7. Undo bị từ chối không làm thay đổi dữ liệu và trạng thái
    def test_undo_rejected_does_not_modify_state(self) -> None:
        mgr = VectorEngineManager()
        mgr.undo_stack = MockLinkedStackWithPeek()

        mgr.add_record("cat", [1.0, 2.0], record_id="C1")
        mgr.delete_record("cat")
        mgr.add_record("cat", [9.0, 9.0], record_id="C2", track_undo=False)

        records_snapshot = {k: v for k, v in mgr._records.items()}
        label_snapshot = dict(mgr._label_to_id)
        count_snapshot = mgr.record_count
        dim_snapshot = mgr.dimension

        # Undo bị từ chối
        self.assertIsNone(mgr.undo())

        # Mọi trạng thái phải hoàn toàn bất biến
        self.assertEqual(mgr._records, records_snapshot)
        self.assertEqual(mgr._label_to_id, label_snapshot)
        self.assertEqual(mgr.record_count, count_snapshot)
        self.assertEqual(mgr.dimension, dim_snapshot)

    # 8. Undo thành công vẫn tuân thủ LIFO và toàn vẹn hai chiều
    def test_undo_lifo_order_success(self) -> None:
        mgr = VectorEngineManager()
        mgr.undo_stack = MockLinkedStackWithPeek()

        mgr.add_record("w1", [1.0, 1.0])
        mgr.add_record("w2", [2.0, 2.0])
        mgr.add_record("w3", [3.0, 3.0])
        mgr.update_record("w2", [2.5, 2.5])
        mgr.delete_record("w1")

        # 5 lần Undo theo thứ tự ngược lại (LIFO):
        # 1. Undo DELETE w1
        msg1 = mgr.undo()
        self.assertIn("w1", msg1)
        self.assertIsNotNone(mgr.get_record("w1"))

        # 2. Undo UPDATE w2
        msg2 = mgr.undo()
        self.assertIn("w2", msg2)
        self.assertEqual(mgr.get_record("w2").values, [2.0, 2.0])

        # 3. Undo ADD w3
        msg3 = mgr.undo()
        self.assertIn("w3", msg3)
        self.assertIsNone(mgr.get_record("w3"))

        # 4. Undo ADD w2
        msg4 = mgr.undo()
        self.assertIn("w2", msg4)
        self.assertIsNone(mgr.get_record("w2"))

        # 5. Undo ADD w1
        msg5 = mgr.undo()
        self.assertIn("w1", msg5)
        self.assertIsNone(mgr.get_record("w1"))

        # Hết thao tác
        self.assertEqual(mgr.record_count, 0)
        self.assertEqual(mgr.dimension, 0)
        self.assertEqual(len(mgr._records), 0)
        self.assertEqual(len(mgr._label_to_id), 0)
        self.assertIsNone(mgr.undo())

    # 9. Undo UPDATE không tạo orphan record và từ chối nếu bản ghi bị xóa ngoài luồng
    def test_undo_update_integrity_and_refusal_when_deleted(self) -> None:
        mgr = VectorEngineManager()
        mgr.undo_stack = MockLinkedStackWithPeek()

        mgr.add_record("car", [1.0, 1.0], record_id="V_CAR")
        mgr.update_record("car", [2.0, 2.0])

        # Trường hợp 1: Hoàn tác UPDATE bình thường
        undo_msg = mgr.undo()
        self.assertIsNotNone(undo_msg)
        self.assertEqual(mgr.get_record("car").values, [1.0, 1.0])
        self.assertEqual(mgr._label_to_id.get("car"), "V_CAR")
        self.assertEqual(mgr._records.get("V_CAR").values, [1.0, 1.0])

        # Trường hợp 2: Update lại, sau đó xóa 'car' mà không ghi undo
        mgr.update_record("car", [3.0, 3.0])
        mgr.delete_record("car", track_undo=False)

        # Undo UPDATE phải bị từ chối an toàn vì record không còn tồn tại
        self.assertIsNone(mgr.undo())
        self.assertEqual(mgr.record_count, 0)
        self.assertEqual(len(mgr._records), 0)
        self.assertEqual(len(mgr._label_to_id), 0)


if __name__ == "__main__":
    unittest.main()
