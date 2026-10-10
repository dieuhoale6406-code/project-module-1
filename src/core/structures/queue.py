from typing import Any, Optional


class QueueNode:
    """Nút liên kết cho hàng đợi."""

    def __init__(self, data: Any, next_node: Optional["QueueNode"] = None) -> None:
        self.data = data
        self.next = next_node


class LinkedQueue:
    """
    Khung sườn LinkedQueue.
    Nhiệm vụ: Cài đặt hàng đợi tự xây dựng bằng danh sách liên kết với 2 con trỏ front và rear.
    Mục đích: Phục vụ hàng đợi tác vụ truy vấn vector dạng batch.
    """

    def __init__(self) -> None:
        self.front: Optional[QueueNode] = None
        self.rear: Optional[QueueNode] = None
        self._size: int = 0

    def is_empty(self) -> bool:
        return self.front is None

    def size(self) -> int:
        return self._size

    def enqueue(self, item: Any) -> None:
        """Thêm phần tử vào cuối hàng đợi."""
        raise NotImplementedError("Cài đặt thao tác enqueue.")

    def dequeue(self) -> Any:
        """Lấy phần tử ra khỏi đầu hàng đợi."""
        raise NotImplementedError("Cài đặt thao tác dequeue.")

    def peek(self) -> Optional[Any]:
        """Xem phần tử ở đầu hàng đợi mà không lấy ra."""
        raise NotImplementedError("Cài đặt thao tác peek.")
