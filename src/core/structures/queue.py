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
        new_node = QueueNode(item)

        if self.rear is None:
            self.front = new_node
        else:
            self.rear.next = new_node

        self.rear = new_node
        self._size += 1

    def dequeue(self) -> Any:
        if self.front is None:
            raise IndexError("Dequeue from empty queue.")

        item = self.front.data
        self.front = self.front.next

        if self.front is None:
            self.rear = None

        self._size -= 1
        return item

    def peek(self) -> Optional[Any]:
        if self.front is None:
            return None

        return self.front.data

    def remove_rear(self) -> Any:
        """Xóa và trả về phần tử ở cuối hàng đợi."""
        if self.front is None:
            raise IndexError("Dequeue from empty queue.")

        if self.front is self.rear:
            item = self.front.data
            self.front = None
            self.rear = None
            self._size = 0
            return item

        current = self.front
        while current.next is not self.rear and current.next is not None:
            current = current.next

        item = self.rear.data
        current.next = None
        self.rear = current
        self._size -= 1
        return item

    def dequeue_rear(self) -> Any:
        """Bí danh cho remove_rear."""
        return self.remove_rear()

