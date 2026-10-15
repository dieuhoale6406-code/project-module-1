from typing import Any, Optional


class StackNode:
    """Nút liên kết cho ngăn xếp."""

    def __init__(self, data: Any, next_node: Optional["StackNode"] = None) -> None:
        self.data = data
        self.next = next_node


class LinkedStack:
    """
    Nhiệm vụ: Cài đặt ngăn xếp tự xây dựng bằng danh sách liên kết đơn (Linked List).
    Mục đích: Lưu trữ lịch sử thao tác để phục vụ chức năng Undo.
    """

    def __init__(self) -> None:
        self.top: Optional[StackNode] = None
        self._size: int = 0

    def is_empty(self) -> bool:
        return self.top is None

    def size(self) -> int:
        return self._size

    def push(self, item: Any) -> None:
        """Đẩy một phần tử vào đỉnh ngăn xếp."""
        new_node = StackNode(item, self.top)
        self.top = new_node
        self._size += 1

    def pop(self) -> Any:
        """Lấy phần tử ra khỏi đỉnh ngăn xếp."""
        if self.is_empty():
            raise IndexError("Pop from empty stack.")
        item = self.top.data
        self.top = self.top.next
        self._size -= 1
        return item

    def peek(self) -> Optional[Any]:
        """Xem phần tử ở đỉnh ngăn xếp mà không lấy ra."""
        if self.is_empty():
            return None
        return self.top.data
