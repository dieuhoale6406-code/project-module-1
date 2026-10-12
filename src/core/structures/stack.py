from typing import Any, Optional


class StackNode:
    """Nút liên kết cho ngăn xếp."""

    def __init__(self, data: Any, next_node: Optional["StackNode"] = None) -> None:
        self.data = data
        self.next = next_node


class LinkedStack:
    """
    Cài đặt cấu trúc dữ liệu Stack bằng danh sách liên kết đơn.
    Mục đích: Lưu trữ lịch sử thao tác để phục vụ chức năng Undo.
    """

    def __init__(self) -> None:
        """
        Khởi tạo một stack rỗng

        Args:
            top (Optional[StackNode]): Nút đầu stack
            size (int): Số lượng phần tử trong stack
        """
        self.top: Optional[StackNode] = None
        self._size: int = 0

    def is_empty(self) -> bool:
        """
        Kiểm tra xem stack có rỗng không.

        Returns:
            bool: True nếu stack rỗng, False ngược lại.
        """
        return self.top is None

    def size(self) -> int:
        """
        Lấy số lượng phần tử trong stack.

        Returns:
            int: Số lượng phần tử trong stack
        """
        return self._size

    def push(self, item: Any) -> None:
        """
        Thêm một phần tử vào đỉnh stack.

        Args:
            item (Any): Phần tử cần thêm.
        """
        new_node = StackNode(item, self.top)
        self.top = new_node
        self._size += 1

    def pop(self) -> Any:
        """
        Lấy phần tử ra khỏi đỉnh ngăn xếp.

        Returns:
            Any: Phần tử ở đỉnh stack.
        
        Raises:
            IndexError: Nếu stack rỗng.
        """
        if self.is_empty():
            raise IndexError("Không thể pop từ stack rỗng")

        popped_node = self.top
        self.top = popped_node.next
        self._size -= 1
        return popped_node.data

    def peek(self) -> Optional[Any]:
        """
        Xem phần tử ở đỉnh ngăn xếp mà không lấy ra.

        Returns:
            Optional[Any]: Phần tử ở đỉnh stack nếu không rỗng, None ngược lại.
        """
        if self.is_empty():
            return None
        return self.top.data

    def clear(self) -> None:
        """
        Làm rỗng stack.
        """
        self.top = None
        self._size = 0 