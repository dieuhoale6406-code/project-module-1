from typing import Any, List, Optional, Tuple


class BoundedMaxHeap:
    """
    Khung sườn cấu trúc Bounded Max-Heap.
    Nhiệm vụ: Hiện thực Heap bằng mảng 1D (list) thuần, tự code phép toán sift-up và sift-down.
    Mục đích: Duy trì Top-K khoảng cách nhỏ nhất với độ phức tạp chèn O(log K).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Mảng lưu trữ các phần tử dạng tuple (distance, item)
        self._heap: List[Tuple[float, Any]] = []

    def size(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def peek_max(self) -> Optional[Tuple[float, Any]]:
        """Lấy phần tử lớn nhất (ở đỉnh heap) mà không lấy ra."""
        if self.is_empty():
            return None
        return self._heap[0]

    def _sift_up(self, index: int) -> None:
        """Đẩy phần tử tại index lên trên để bảo toàn tính chất Max-Heap."""
        raise NotImplementedError("Cài đặt thao tác _sift_up.")

    def _sift_down(self, index: int) -> None:
        """Hạ phần tử tại index xuống dưới để bảo toàn tính chất Max-Heap."""
        raise NotImplementedError("Cài đặt thao tác _sift_down.")

    def push(self, distance: float, item: Any) -> None:
        """
        Thêm một phần tử vào heap:
        - Nếu heap chưa đầy: thêm vào cuối và _sift_up.
        - Nếu heap đã đầy và distance < distance ở đỉnh: thay thế đỉnh và _sift_down.
        """
        raise NotImplementedError("Cài đặt thao tác push.")

    def get_sorted_results(self) -> List[Tuple[Any, float]]:
        """Lấy toàn bộ kết quả đã sắp xếp theo khoảng cách tăng dần."""
        raise NotImplementedError("Cài đặt thao tác lấy danh sách kết quả.")
