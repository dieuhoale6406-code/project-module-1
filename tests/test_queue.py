import sys
from pathlib import Path
import unittest

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.structures.queue import LinkedQueue


class TestLinkedQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = LinkedQueue()

    def test_initial_state_is_empty(self) -> None:
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)
        self.assertIsNone(self.queue.front)
        self.assertIsNone(self.queue.rear)

    def test_peek_on_empty_queue_returns_none(self) -> None:
        self.assertIsNone(self.queue.peek())
        self.assertEqual(self.queue.size(), 0)

    def test_dequeue_on_empty_queue_raises_without_changing_state(self) -> None:
        for _ in range(2):
            with self.assertRaises(IndexError):
                self.queue.dequeue()
            self.assertTrue(self.queue.is_empty())
            self.assertEqual(self.queue.size(), 0)
            self.assertIsNone(self.queue.front)
            self.assertIsNone(self.queue.rear)

    def test_enqueue_single_item(self) -> None:
        self.queue.enqueue(42)
        self.assertFalse(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 1)
        self.assertEqual(self.queue.peek(), 42)
        self.assertIsNotNone(self.queue.front)
        self.assertIs(self.queue.front, self.queue.rear)
        self.assertIsNone(self.queue.rear.next)

    def test_enqueue_increments_size_and_preserves_front(self) -> None:
        for size, item in enumerate([10, 20, 30], start=1):
            self.queue.enqueue(item)
            self.assertEqual(self.queue.size(), size)
            self.assertEqual(self.queue.peek(), 10)
            self.assertEqual(self.queue.rear.data, item)
            self.assertIsNone(self.queue.rear.next)

    def test_dequeue_returns_fifo_order_and_decrements_size(self) -> None:
        items = [10, 20, 30, 40]
        for item in items:
            self.queue.enqueue(item)
        for index, expected in enumerate(items):
            self.assertEqual(self.queue.dequeue(), expected)
            self.assertEqual(self.queue.size(), len(items) - index - 1)
        self.assertTrue(self.queue.is_empty())

    def test_dequeue_last_item_resets_both_pointers(self) -> None:
        self.queue.enqueue("only")
        self.assertEqual(self.queue.dequeue(), "only")
        self.assertIsNone(self.queue.front)
        self.assertIsNone(self.queue.rear)
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)
        self.assertIsNone(self.queue.peek())
        with self.assertRaises(IndexError):
            self.queue.dequeue()

    def test_peek_does_not_remove_item(self) -> None:
        self.queue.enqueue("first")
        self.queue.enqueue("second")
        for _ in range(3):
            self.assertEqual(self.queue.peek(), "first")
            self.assertEqual(self.queue.size(), 2)
        self.assertEqual(self.queue.dequeue(), "first")
        self.assertEqual(self.queue.peek(), "second")
        self.assertEqual(self.queue.dequeue(), "second")

    def test_none_is_a_valid_item(self) -> None:
        self.queue.enqueue(None)
        self.assertFalse(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 1)
        self.assertIsNone(self.queue.peek())
        self.assertIsNone(self.queue.dequeue())
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)

    def test_various_types_preserve_object_identity(self) -> None:
        items = [0, False, "", 3.14, [1, 2], {"key": "value"}, (1, 2)]
        for item in items:
            self.queue.enqueue(item)
        for item in items:
            self.assertIs(self.queue.peek(), item)
            self.assertIs(self.queue.dequeue(), item)

    def test_duplicate_values_are_kept(self) -> None:
        for item in [7, 7, 8, 7]:
            self.queue.enqueue(item)
        self.assertEqual(self.queue.size(), 4)
        self.assertEqual([self.queue.dequeue() for _ in range(4)], [7, 7, 8, 7])
        self.assertTrue(self.queue.is_empty())

    def test_enqueue_and_dequeue_interleaved(self) -> None:
        self.queue.enqueue("A")
        self.queue.enqueue("B")
        self.assertEqual(self.queue.dequeue(), "A")
        self.queue.enqueue("C")
        self.assertEqual(self.queue.dequeue(), "B")
        self.queue.enqueue("D")
        self.assertEqual(self.queue.size(), 2)
        self.assertEqual(self.queue.dequeue(), "C")
        self.assertEqual(self.queue.dequeue(), "D")
        self.assertTrue(self.queue.is_empty())

    def test_reuse_after_draining_queue(self) -> None:
        for cycle in range(3):
            items = [cycle, cycle + 10]
            for item in items:
                self.queue.enqueue(item)
            self.assertEqual(self.queue.size(), 2)
            for item in items:
                self.assertEqual(self.queue.dequeue(), item)
            self.assertTrue(self.queue.is_empty())
            self.assertEqual(self.queue.size(), 0)
            self.assertIsNone(self.queue.front)
            self.assertIsNone(self.queue.rear)

    def test_queue_instances_are_independent(self) -> None:
        other = LinkedQueue()
        self.queue.enqueue("first queue")
        self.assertTrue(other.is_empty())
        other.enqueue("second queue")
        self.assertEqual(self.queue.dequeue(), "first queue")
        self.assertEqual(other.size(), 1)
        self.assertEqual(other.dequeue(), "second queue")

    def test_large_batch_preserves_fifo_order(self) -> None:
        count = 10_000
        for item in range(count):
            self.queue.enqueue(item)
        self.assertEqual(self.queue.size(), count)
        for expected in range(count):
            self.assertEqual(self.queue.dequeue(), expected)
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
