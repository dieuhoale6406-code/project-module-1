import sys
from pathlib import Path
import unittest

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.structures.stack import LinkedStack, StackNode


class TestLinkedStackInit(unittest.TestCase):
    """Kiem tra khoi tao LinkedStack."""

    def test_initial_state_is_empty(self) -> None:
        """Stack moi phai rong, size = 0, top = None."""
        stack = LinkedStack()
        self.assertTrue(stack.is_empty())
        self.assertEqual(stack.size(), 0)
        self.assertIsNone(stack.top)

    def test_peek_on_empty_stack_returns_none(self) -> None:
        """peek() tren stack rong tra ve None."""
        stack = LinkedStack()
        self.assertIsNone(stack.peek())

    def test_initial_size_is_zero(self) -> None:
        """_size ban dau phai la 0."""
        stack = LinkedStack()
        self.assertEqual(stack._size, 0)


class TestLinkedStackPush(unittest.TestCase):
    """Kiem tra thao tac push."""

    def test_push_single_item(self) -> None:
        stack = LinkedStack()
        stack.push(42)
        self.assertFalse(stack.is_empty())
        self.assertEqual(stack.size(), 1)
        self.assertEqual(stack.peek(), 42)

    def test_push_multiple_items_lifo_order(self) -> None:
        stack = LinkedStack()
        stack.push("first")
        stack.push("second")
        stack.push("third")
        self.assertEqual(stack.size(), 3)
        self.assertEqual(stack.peek(), "third")

    def test_push_increments_size(self) -> None:
        stack = LinkedStack()
        for i in range(1, 6):
            stack.push(i)
            self.assertEqual(stack.size(), i)

    def test_push_none_value(self) -> None:
        stack = LinkedStack()
        stack.push(None)
        self.assertEqual(stack.size(), 1)
        self.assertIsNone(stack.peek())

    def test_push_various_types(self) -> None:
        stack = LinkedStack()
        items = [1, 3.14, "text", [1, 2], {"key": "val"}, (1, 2)]
        for item in items:
            stack.push(item)
        self.assertEqual(stack.size(), len(items))
        self.assertEqual(stack.peek(), items[-1])

    def test_push_updates_top_pointer(self) -> None:
        stack = LinkedStack()
        stack.push("a")
        stack.push("b")
        self.assertIsInstance(stack.top, StackNode)
        self.assertEqual(stack.top.data, "b")
        self.assertIsNotNone(stack.top.next)
        self.assertEqual(stack.top.next.data, "a")


class TestLinkedStackPop(unittest.TestCase):
    """Kiem tra thao tac pop."""

    def test_pop_single_item(self) -> None:
        stack = LinkedStack()
        stack.push(99)
        result = stack.pop()
        self.assertEqual(result, 99)
        self.assertTrue(stack.is_empty())
        self.assertEqual(stack.size(), 0)

    def test_pop_returns_lifo_order(self) -> None:
        stack = LinkedStack()
        items = [10, 20, 30, 40]
        for item in items:
            stack.push(item)
        for expected in reversed(items):
            self.assertEqual(stack.pop(), expected)
        self.assertTrue(stack.is_empty())

    def test_pop_decrements_size(self) -> None:
        stack = LinkedStack()
        n = 5
        for i in range(n):
            stack.push(i)
        for remaining in range(n - 1, -1, -1):
            stack.pop()
            self.assertEqual(stack.size(), remaining)

    def test_pop_from_empty_stack_raises_index_error(self) -> None:
        stack = LinkedStack()
        with self.assertRaises(IndexError):
            stack.pop()

    def test_pop_message_on_empty(self) -> None:
        stack = LinkedStack()
        with self.assertRaises(IndexError) as ctx:
            stack.pop()
        self.assertIn("empty stack", str(ctx.exception).lower())

    def test_pop_after_empty_raises_again(self) -> None:
        stack = LinkedStack()
        stack.push("x")
        stack.pop()
        with self.assertRaises(IndexError):
            stack.pop()

    def test_pop_updates_top_pointer(self) -> None:
        stack = LinkedStack()
        stack.push("a")
        stack.push("b")
        stack.pop()
        self.assertEqual(stack.top.data, "a")
        self.assertIsNone(stack.top.next)


class TestLinkedStackPeek(unittest.TestCase):
    """Kiem tra thao tac peek."""

    def test_peek_does_not_remove_item(self) -> None:
        stack = LinkedStack()
        stack.push("hello")
        before_size = stack.size()
        val = stack.peek()
        self.assertEqual(val, "hello")
        self.assertEqual(stack.size(), before_size)

    def test_peek_returns_top_element(self) -> None:
        stack = LinkedStack()
        stack.push(1)
        self.assertEqual(stack.peek(), 1)
        stack.push(2)
        self.assertEqual(stack.peek(), 2)
        stack.push(3)
        self.assertEqual(stack.peek(), 3)

    def test_peek_after_pop(self) -> None:
        stack = LinkedStack()
        stack.push("a")
        stack.push("b")
        stack.pop()
        self.assertEqual(stack.peek(), "a")

    def test_peek_on_empty_returns_none(self) -> None:
        stack = LinkedStack()
        self.assertIsNone(stack.peek())


class TestLinkedStackIsEmpty(unittest.TestCase):
    """Kiem tra thao tac is_empty."""

    def test_is_empty_on_new_stack(self) -> None:
        self.assertTrue(LinkedStack().is_empty())

    def test_is_empty_false_after_push(self) -> None:
        stack = LinkedStack()
        stack.push(1)
        self.assertFalse(stack.is_empty())

    def test_is_empty_true_after_all_pops(self) -> None:
        stack = LinkedStack()
        stack.push(1)
        stack.push(2)
        stack.pop()
        stack.pop()
        self.assertTrue(stack.is_empty())


class TestLinkedStackIntegration(unittest.TestCase):
    """Kiem tra tich hop va kich ban thuc te."""

    def test_push_pop_interleaved(self) -> None:
        stack = LinkedStack()
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.pop(), 2)
        stack.push(3)
        stack.push(4)
        self.assertEqual(stack.pop(), 4)
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.pop(), 1)
        self.assertTrue(stack.is_empty())

    def test_large_push_pop(self) -> None:
        stack = LinkedStack()
        n = 10_000
        for i in range(n):
            stack.push(i)
        self.assertEqual(stack.size(), n)
        for i in range(n - 1, -1, -1):
            self.assertEqual(stack.pop(), i)
        self.assertTrue(stack.is_empty())

    def test_undo_history_simulation(self) -> None:
        history = LinkedStack()
        operations = ["add A", "add B", "add C", "add D"]
        for op in operations:
            history.push(op)
        self.assertEqual(history.size(), 4)
        undone = []
        while not history.is_empty():
            undone.append(history.pop())
        self.assertEqual(undone, list(reversed(operations)))

    def test_reuse_after_clearing_via_pop(self) -> None:
        stack = LinkedStack()
        for i in range(5):
            stack.push(i)
        while not stack.is_empty():
            stack.pop()
        stack.push("new")
        self.assertEqual(stack.size(), 1)
        self.assertEqual(stack.peek(), "new")
        self.assertEqual(stack.pop(), "new")
        self.assertTrue(stack.is_empty())

    def test_linked_structure_integrity(self) -> None:
        stack = LinkedStack()
        for val in [1, 2, 3]:
            stack.push(val)
        node = stack.top
        expected = [3, 2, 1]
        for exp in expected:
            self.assertIsNotNone(node)
            self.assertEqual(node.data, exp)
            node = node.next
        self.assertIsNone(node)


if __name__ == "__main__":
    unittest.main()
