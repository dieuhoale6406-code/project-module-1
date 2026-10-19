import math
import os
from pathlib import Path
import sys
import tempfile
from typing import List
import unittest

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.manager import VectorEngineManager
from core.storage import (
    TextVectorStorage,
    _parse_lines_chunk,
    get_optimal_worker_count,
)


class TestStorageMultiprocessing(unittest.TestCase):
    """Kiểm thử tính năng nạp dữ liệu đa tiến trình và thích ứng cấu hình phần cứng."""

    def test_get_optimal_worker_count(self) -> None:
        """Kiểm tra thuật toán phân bổ worker thích ứng với CPU."""
        # 1 lõi hoặc 0 lõi -> tối thiểu 1 worker
        self.assertEqual(get_optimal_worker_count(0), 1)
        self.assertEqual(get_optimal_worker_count(1), 1)
        # 2 lõi -> dùng 1 worker (để lại 1 lõi cho OS và Streamlit)
        self.assertEqual(get_optimal_worker_count(2), 1)
        # 4 lõi -> dùng 3 workers
        self.assertEqual(get_optimal_worker_count(4), 3)
        # 8 lõi -> dùng 4 hoặc 6 workers (giới hạn an toàn, không chiếm 100% CPU)
        w8 = get_optimal_worker_count(8)
        self.assertGreaterEqual(w8, 4)
        self.assertLessEqual(w8, 6)
        # 16 lõi -> tối đa 6 đến 8 workers
        w16 = get_optimal_worker_count(16)
        self.assertLessEqual(w16, 8)

    def test_parse_lines_chunk(self) -> None:
        """Kiểm tra worker parse chuỗi độc lập."""
        lines = [
            "apple 0.1 0.2 0.3\n",
            "# this is a comment\n",
            "   \n",
            "new york city 0.4 0.5 0.6\n",
            "bad_length 0.1 0.2\n",
            "bad_number 0.1 abc 0.3\n",
            "nan_val 0.1 nan 0.3\n",
        ]
        parsed = _parse_lines_chunk(lines, expected_dim=3)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0][0], "apple")
        self.assertEqual(parsed[0][1], [0.1, 0.2, 0.3])
        self.assertEqual(parsed[1][0], "new york city")
        self.assertEqual(parsed[1][1], [0.4, 0.5, 0.6])

    def test_load_vectors_small_file_single_thread(self) -> None:
        """File nhỏ hoặc limit nhỏ tự động chạy single-thread để tránh overhead tạo process."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            for i in range(50):
                f.write(f"word_{i} 0.1 0.2 0.3\n")
            temp_path = f.name

        try:
            storage = TextVectorStorage(file_path=temp_path, expected_dim=3)
            # limit = 20 <= 5000: chạy in-process
            results = storage.load_vectors(limit=20)
            self.assertEqual(len(results), 20)
            self.assertEqual(results[0][0], "word_0")
            self.assertEqual(results[19][0], "word_19")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_vectors_multiprocess_execution(self) -> None:
        """Kiểm tra nạp dữ liệu bằng chế độ đa tiến trình (force_multiprocessing=True)."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            for i in range(200):
                f.write(f"item_{i:04d} 1.0 2.0 3.0\n")
            temp_path = f.name

        try:
            storage = TextVectorStorage(file_path=temp_path, expected_dim=3)
            results = storage.load_vectors(force_multiprocessing=True, chunk_size=50)
            self.assertEqual(len(results), 200)
            self.assertEqual(results[0][0], "item_0000")
            self.assertEqual(results[199][0], "item_0199")
            self.assertEqual(results[0][1], [1.0, 2.0, 3.0])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_manager_add_records_batch(self) -> None:
        """Kiểm tra phương thức add_records_batch nạp hàng loạt vào manager siêu nhanh."""
        manager = VectorEngineManager()
        items = [(f"word_{i}", [float(i), float(i * 2)]) for i in range(100)]
        # Thêm 1 từ trùng để test bỏ qua duplicate
        items.append(("word_0", [99.0, 99.0]))

        loaded, skipped = manager.add_records_batch(items)
        self.assertEqual(loaded, 100)
        self.assertEqual(skipped, 1)
        self.assertEqual(manager.record_count, 100)
        self.assertEqual(manager.dimension, 2)
        self.assertIsNotNone(manager.get_record("word_0"))
        self.assertIsNotNone(manager.get_record("word_99"))


if __name__ == "__main__":
    unittest.main()
