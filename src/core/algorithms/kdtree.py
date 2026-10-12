import heapq
import math
from typing import Any, List, Optional, Tuple

from core.metrics import euclidean_distance
from .base import BaseVectorIndex


class KDTreeNode:
    """
    Nút trên cây phân vùng không gian đa chiều KD-Tree.
    
    Thuộc tính:
        record: Đối tượng VectorRecord lưu tại nút này.
        left: Con trỏ tới nút con bên trái (chứa các điểm có tọa độ tại axis nhỏ hơn).
        right: Con trỏ tới nút con bên phải (chứa các điểm có tọa độ tại axis lớn hơn hoặc bằng).
        axis: Trục tọa độ được chọn để phân chia tại tầng hiện tại (0 <= axis < dimension).
    """

    def __init__(
        self,
        record: Any,
        left: Optional["KDTreeNode"] = None,
        right: Optional["KDTreeNode"] = None,
        axis: int = 0
    ) -> None:
        self.record = record
        self.left = left
        self.right = right
        self.axis = axis


class KDTreeIndex(BaseVectorIndex):
    """
    Thuật toán 2: K-Dimensional Tree (Bentley, 1975)
    Độ phức tạp: Build O(N log N * d), Search O(log N * d) ở số chiều thấp.
    Cơ chế: Phân vùng trực giao theo trục tọa độ axis = depth % dimension.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(name="KD-Tree")
        self.dimension = dimension
        self.root: Optional[KDTreeNode] = None

    def _build_recursive(self, records: List[Any], depth: int = 0) -> Optional[KDTreeNode]:
        # Điều kiện dừng khi không còn vector
        if not records:
            return None

        # Chọn trục phân chia
        axis = depth % self.dimension

        sorted_records = sorted(
            records,
            key=lambda record: record.values[axis]
        )

        # Tính median
        median_idx = len(sorted_records) // 2
        median_record = sorted_records[median_idx]

        # Tạo Node
        node = KDTreeNode(median_record, axis=axis)

        # Xây dựng 2 cây con
        node.left = self._build_recursive(
            sorted_records[:median_idx],
            depth + 1
        )

        node.right = self._build_recursive(
            sorted_records[median_idx + 1:],
            depth + 1
        )
        
        return node

    def build(self, records: List[Any]) -> None:
        """
        Khởi tạo và dựng cây KD-Tree từ tập bản ghi đầu vào.
        
        Tham số:
            records: Danh sách VectorRecord.
        """
        if self.dimension <= 0:
            raise ValueError("Dimension must be greater than 0")

        for record in records:
            if len(record.values) != self.dimension:
                raise ValueError(
                    f"Record {record.id} has dimension"
                    f"{len(record.values)}, expected {self.dimension}"
                )

            for value in record.values:
                if(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Record {record.id} contains "
                        f"an invalid coordinate: {value!r}"
                    )

        self.root = self._build_recursive(records, depth=0)

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector gần nhất bằng Euclidean Distance với kỹ thuật Pruning.

        Trả về: Danh sách [(VectorRecord, khoảng_cách), ...] sắp xếp tăng dần theo khoảng cách.
        """
        if self.root is None or top_k <= 0:
            return []

        if len(query_values) != self.dimension:
            raise ValueError(
                f"Query vector có {len(query_values)} chiều, kỳ vọng {self.dimension} chiều."
            )

        # Sử dụng entry_counter để tránh so sánh VectorRecord khi trùng khoảng cách
        max_heap: List[Tuple[float, int, Any]] = []
        entry_counter = 0

        def _search_node(node: Optional[KDTreeNode]) -> None:
            nonlocal entry_counter
            if node is None:
                return

            dist = euclidean_distance(query_values, node.record.values)

            # Cập nhật Max-Heap duy trì Top-K
            if len(max_heap) < top_k:
                entry_counter += 1
                heapq.heappush(max_heap, (-dist, entry_counter, node.record))
            elif dist < -max_heap[0][0]:
                entry_counter += 1
                heapq.heapreplace(max_heap, (-dist, entry_counter, node.record))

            # Xác định trục phân cách và khoảng cách chiếu tới siêu phẳng
            axis = node.axis
            diff = query_values[axis] - node.record.values[axis]

            # Nhánh ưu tiên (near) và nhánh đối diện (far)
            if diff < 0:
                near_branch = node.left
                far_branch = node.right
            else:
                near_branch = node.right
                far_branch = node.left

            # Luôn duyệt nhánh gần trước
            _search_node(near_branch)
            
            # Chỉ duyệt nhánh xa nếu Heap chưa đủ K phần tử HOẶC khoảng cách |diff| < khoảng cách tệ nhất hiện tại
            worst_dist = -max_heap[0][0] if max_heap else float("inf")
            if len(max_heap) < top_k or abs(diff) < worst_dist:
                _search_node(far_branch)

        _search_node(self.root)

        # Trích xuất từ heap và sắp xếp tăng dần theo khoảng cách thực (dương)
        results = [(-neg_dist, record) for neg_dist, _, record in max_heap]
        results.sort(key=lambda x: x[0])
        return [(record, dist) for dist, record in results]
