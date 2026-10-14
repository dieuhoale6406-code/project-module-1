from collections.abc import Callable
from typing import Any, List, Optional, Tuple
from .base import BaseVectorIndex
from core.metrics import euclidean_distance
from core.structures import BoundedMaxHeap
import math
import random


class KDTreeNode:
    """
    Nút trên cây phân vùng không gian đa chiều KD-Tree.

    Thuộc tính:
        record: Đối tượng VectorRecord lưu tại nút này.
        left: Cây con chứa các điểm có tọa độ tại axis <= tọa độ nút.
        right: Cây con chứa các điểm có tọa độ tại axis >= tọa độ nút.
        axis: Trục phân chia tại tầng hiện tại.
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
    KD-Tree sử dụng Quickselect để xây dựng cây và BoundedMaxHeap
    để tìm kiếm Top-K láng giềng gần nhất.

    Build:
        Expected O(N log N * d) với Quickselect ngẫu nhiên.

    Search:
        O(V * (d + log K) + K log K), với V là số node được thăm.
        Trong trường hợp xấu nhất, V = N.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(name="KD-Tree")

        self.dimension = dimension
        self.root: Optional[KDTreeNode] = None
        self._size = 0

    def _quickselect(
        self,
        records: List[Any],
        left: int,
        right: int,
        k: int,
        key: Callable[[Any], Any],
    ) -> Any:
        """
        Đưa phần tử có thứ hạng k vào đúng vị trí phân hoạch,
        không cần sắp xếp toàn bộ đoạn dữ liệu.

        Các phần tử bên trái có key <= key(records[k]).
        Các phần tử bên phải có key >= key(records[k]).

        Độ phức tạp kỳ vọng: O(N) trên đoạn đang xét.
        """
        while left < right:
            pivot_idx = random.randint(left, right)
            pivot = key(records[pivot_idx])

            # Three-way partition:
            # [left, lt - 1] < pivot
            # [lt, gt]       == pivot
            # [gt + 1, right] > pivot
            lt = left
            i = left
            gt = right

            while i <= gt:
                value = key(records[i])
                if value < pivot:
                    records[lt], records[i] = (records[i], records[lt])
                    lt += 1
                    i += 1
                elif value > pivot:
                    records[i], records[gt] = (records[gt], records[i])
                    gt -= 1
                else:
                    i += 1
            
            if k < lt:
                right = lt - 1
            elif k > gt:
                left = gt + 1
            else:
                return records[k]
        
        return records[left]
        

    def _build_recursive(
        self,
        records: List[Any],
        left: int,
        right: int,
        depth: int = 0,
    ) -> Optional[KDTreeNode]:
        """
        Xây dựng KD-Tree bằng Quickselect.

        Sử dụng chỉ số left/right thay vì slicing để tránh
        tạo nhiều danh sách con không cần thiết.
        """

        if left > right:
            return None

        # Chọn trục phân chia
        axis = depth % self.dimension

        # Tính median
        median_idx = (left + right) // 2

        # Đưa median về đúng vị trí phân hoạch
        self._quickselect(
            records,
            left,
            right,
            median_idx,
            key=lambda record: record.values[axis],
        )

        # Tạo Node
        node = KDTreeNode(
            record=records[median_idx], 
            axis=axis
        )

        # Xây dựng 2 cây con
        node.left = self._build_recursive(
            records,
            left,
            median_idx - 1,
            depth + 1,
        )

        node.right = self._build_recursive(
            records,
            median_idx + 1,
            right,
            depth + 1,
        )
        
        return node

    def build(self, records: List[Any]) -> None:
        """
        Xây dựng KD-Tree từ danh sách VectorRecord.

        Không thay đổi thứ tự của danh sách records đầu vào.
        """
        # Validate dimension.
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
        ):
            raise TypeError("Dimension must be an integer.")

        if self.dimension <= 0:
            raise ValueError("Dimension must be greater than 0.")

        if not isinstance(records, list):
            raise TypeError("records must be a list.")

        # Validate toàn bộ dữ liệu trước khi xây cây.
        for record in records:
            if not hasattr(record, "values"):
                raise ValueError(
                    "Each record must have a values attribute."
                )

            if len(record.values) != self.dimension:
                record_id = getattr(record, "id", "<unknown>")
                raise ValueError(
                    f"Record {record_id} has dimension "
                    f"{len(record.values)}, expected "
                    f"{self.dimension}."
                )

            for value in record.values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    record_id = getattr(record, "id", "<unknown>")
                    raise ValueError(
                        f"Record {record_id} contains an invalid "
                        f"coordinate: {value!r}."
                    )

        # Phân vùng không làm thay đổi danh sách input
        working_records = records.copy()

        new_root = self._build_recursive(
            working_records,
            left=0,
            right=len(working_records) - 1,
            depth=0,
        )

        self.root = new_root
        self._size = len(working_records)


    def search(
        self, 
        query_values: List[float], 
        top_k: int
    ) -> List[Tuple[Any, float]]:

        # Validation top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # Vaidation query
        if not isinstance(query_values, (list, tuple)):
            raise TypeError(
                "query_values must be a list or tuple."
            )

        if len(query_values) != self.dimension:
            raise ValueError(
                f"Query dimension must be {self.dimension}, "
                f"got {len(query_values)}."
            )

        query: List[float] = []

        for i, value in enumerate(query_values):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise TypeError(
                    f"query_values[{i}] must be an int or float."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"query_values[{i}] must be finite."
                )

            query.append(float(value))

        # Nếu cây trống thì trả về rỗng.
        if self.root is None or self._size == 0:
            return []

        k = min(top_k, self._size)
        heap = BoundedMaxHeap(capacity=k)

        def _search_node(
            node: Optional[KDTreeNode],
        ) -> None:

            if node is None: return

            record = node.record

            # Tính khoảng cách — bỏ qua _validate_vectors()
            # vì query và records đã được validate trước khi search.
            dist = euclidean_distance(query, record.values, validate=False)

            # Dùng ID thực của record để tie-breaking nhất quán,
            # không phụ thuộc vào thứ tự duyệt cây.
            record_id = str(getattr(record, "id", id(record)))

            heap.push(
                distance=dist,
                item=record,
                item_id=record_id
            )

            # Xác định nhánh gần và xa
            axis = node.axis
            diff = query[axis] - record.values[axis]

            # Lấy 2 nhánh
            if diff < 0:
                near_node = node.left
                far_node = node.right
            else:
                near_node = node.right
                far_node = node.left

            # Ưu tiên nhánh gần query hơn
            _search_node(near_node)

            # Xét nhánh xa hơn
            if not heap.is_full():
                _search_node(far_node)
                return

            max_dist = heap.max_distance

            # Plane pruning 
            if abs(diff) < max_dist:
                _search_node(far_node)

        # Bắt đầu tìm kiếm từ root
        _search_node(self.root)

        return heap.get_sorted_results()