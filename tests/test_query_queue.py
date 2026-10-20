import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.manager import VectorEngineManager


APP = """
import streamlit as st
from core.manager import VectorEngineManager
from ui.search import render_search_tab

if "manager" not in st.session_state:
    manager = VectorEngineManager()
    manager.add_record("apple", [1.0, 0.0], track_undo=False)
    manager.add_record("banana", [0.0, 1.0], track_undo=False)
    manager.add_record("cherry", [0.5, 0.5], track_undo=False)
    st.session_state["manager"] = manager
render_search_tab(st.session_state["manager"])
"""


class TestQueryQueueUI(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_string(APP).run()
        self.assertFalse(self.app.exception)

    def click(self, label: str) -> None:
        next(button for button in self.app.button if button.label == label).click().run()
        self.assertFalse(self.app.exception)

    def enqueue(self, word: str) -> None:
        self.app.text_input[0].set_value(word)
        self.click("Thêm vào hàng đợi")

    def test_fifo_one_then_all_with_real_search_results(self) -> None:
        for word in ["banana", "apple", "cherry"]:
            self.enqueue(word)
        queue = self.app.session_state["query_queue"]
        self.assertEqual(queue.size(), 3)
        self.assertEqual(queue.peek()["query_name"], "banana")
        self.assertEqual(queue.rear.data["query_name"], "cherry")
        self.assertEqual(self.app.session_state["query_queue_results"], [])

        self.click("Xử lý truy vấn đầu")
        self.assertEqual(queue.size(), 2)
        self.assertEqual(queue.peek()["query_name"], "apple")
        self.click("Xử lý tất cả")
        history = self.app.session_state["query_queue_results"]
        self.assertEqual([job["query_name"] for job in history], ["banana", "apple", "cherry"])
        for job in history:
            self.assertEqual(job["status"], "Hoàn thành")
            self.assertEqual(job["matches"][0][0].label, job["query_name"])
            self.assertAlmostEqual(job["matches"][0][1], 0.0)
        self.assertTrue(queue.is_empty())
        self.assertIsNone(queue.rear)
        for button in self.app.button:
            if button.label in ("Xử lý truy vấn đầu", "Xử lý tất cả"):
                self.assertTrue(button.disabled)

        self.enqueue("apple")
        self.click("Xử lý truy vấn đầu")
        self.assertEqual(len(self.app.session_state["query_queue_results"]), 4)

    def test_settings_are_saved_at_enqueue_time(self) -> None:
        next(widget for widget in self.app.slider if widget.label == "Số kết quả (Top-K)").set_value(1)
        self.enqueue("apple")
        next(widget for widget in self.app.selectbox if widget.label == "Thuật toán").set_value("lsh").run()
        next(widget for widget in self.app.slider if widget.label.startswith("Số siêu phẳng")).set_value(4)
        self.enqueue("banana")
        queue = self.app.session_state["query_queue"]
        first, second = queue.peek(), queue.rear.data
        self.assertEqual(first["algorithm"], "linear_scan")
        self.assertEqual(first["top_k"], 1)
        self.assertEqual(first["query_values"], [1.0, 0.0])
        self.assertEqual(second["algorithm"], "lsh")
        self.assertEqual(second["params"]["num_hyperplanes"], 4)
        next(widget for widget in self.app.slider if widget.label.startswith("Số siêu phẳng")).set_value(12).run()
        self.assertEqual(second["params"]["num_hyperplanes"], 4)
        with patch.object(VectorEngineManager, "search", return_value=([], 0.1)) as search:
            self.click("Xử lý tất cả")
        self.assertEqual(search.call_args_list[0].args, ("linear_scan", [1.0, 0.0]))
        self.assertEqual(search.call_args_list[0].kwargs["top_k"], 1)
        self.assertEqual(search.call_args_list[1].kwargs["metric"], second["metric"])
        self.assertEqual(search.call_args_list[1].kwargs["num_hyperplanes"], 4)

    def test_failed_query_is_recorded_and_batch_continues(self) -> None:
        self.enqueue("apple")
        self.enqueue("banana")
        with patch.object(VectorEngineManager, "search", side_effect=[ValueError("Test failure"), ([], 0.1)]):
            self.click("Xử lý tất cả")
        history = self.app.session_state["query_queue_results"]
        self.assertEqual([job["status"] for job in history], ["Lỗi", "Hoàn thành"])
        self.assertIn("Test failure", history[0]["error"])
        self.assertTrue(self.app.session_state["query_queue"].is_empty())

    def test_invalid_word_does_not_enqueue(self) -> None:
        for word in ["", "missing"]:
            self.enqueue(word)
            self.assertTrue(self.app.session_state["query_queue"].is_empty())
            self.assertTrue(self.app.error)

    def test_vector_input_is_queued_and_searched(self) -> None:
        next(widget for widget in self.app.radio if widget.label == "Nguồn truy vấn").set_value("Tọa độ vector").run()
        self.app.text_input[0].set_value("0, 1")
        self.click("Thêm vào hàng đợi")
        self.assertEqual(self.app.session_state["query_queue"].peek()["query_values"], [0.0, 1.0])
        self.click("Xử lý truy vấn đầu")
        self.assertEqual(self.app.session_state["query_queue_results"][0]["matches"][0][0].label, "banana")

    def test_dataset_change_clears_pending_queries_and_results(self) -> None:
        self.enqueue("apple")
        self.enqueue("banana")
        self.click("Xử lý truy vấn đầu")
        self.app.session_state["manager"].update_record("apple", [0.2, 0.8])
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.session_state["query_queue"].is_empty())
        self.assertEqual(self.app.session_state["query_queue_results"], [])
        self.assertTrue(any("Dữ liệu đã thay đổi" in item.value for item in self.app.info))

    def test_clear_rear_only_removes_last_query_and_preserves_earlier(self) -> None:
        for word in ["banana", "apple", "cherry"]:
            self.enqueue(word)
        queue = self.app.session_state["query_queue"]
        self.assertEqual(queue.size(), 3)
        self.assertEqual(queue.peek()["query_name"], "banana")
        self.assertEqual(queue.rear.data["query_name"], "cherry")

        # Click "Xóa cuối hàng đợi" -> only "cherry" should be removed
        self.click("Xóa cuối hàng đợi")
        self.assertEqual(queue.size(), 2)
        self.assertEqual(queue.peek()["query_name"], "banana")
        self.assertEqual(queue.rear.data["query_name"], "apple")

        # Click again -> only "apple" should be removed
        self.click("Xóa cuối hàng đợi")
        self.assertEqual(queue.size(), 1)
        self.assertEqual(queue.peek()["query_name"], "banana")
        self.assertEqual(queue.rear.data["query_name"], "banana")

        # Click again -> "banana" removed, queue empty
        self.click("Xóa cuối hàng đợi")
        self.assertTrue(queue.is_empty())
        self.assertIsNone(queue.front)
        self.assertIsNone(queue.rear)

    def test_clear_pending_preserves_completed_results(self) -> None:
        self.enqueue("apple")
        self.enqueue("banana")
        self.click("Xử lý truy vấn đầu")
        self.click("Xóa cuối hàng đợi")
        self.assertTrue(self.app.session_state["query_queue"].is_empty())
        self.assertEqual(len(self.app.session_state["query_queue_results"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
