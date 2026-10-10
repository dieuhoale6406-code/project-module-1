from typing import Any, List, Optional, Tuple
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
        """
        Xây dựng cây KD-Tree đệ quy.
        
        Các bước thực hiện gợi ý:
            1. Điều kiện dừng: Nếu danh sách `records` rỗng, trả về None.
            2. Xác định trục phân chia tại độ sâu hiện tại: axis = depth % self.dimension.
            3. Sắp xếp danh sách `records` theo giá trị tọa độ tại trục `axis`:
               records.sort(key=lambda r: r.values[axis])
            4. Tìm vị trí trung vị (median): median_idx = len(records) // 2.
            5. Tạo nút gốc hiện tại chứa phần tử trung vị: node = KDTreeNode(records[median_idx], axis=axis).
            6. Đệ quy xây dựng cây con trái:
               node.left = self._build_recursive(records[:median_idx], depth + 1)
            7. Đệ quy xây dựng cây con phải:
               node.right = self._build_recursive(records[median_idx + 1:], depth + 1)
            8. Trả về nút `node`.
        """
        raise NotImplementedError("Cài đặt hàm dựng cây đệ quy KD-Tree.")

    def build(self, records: List[Any]) -> None:
        """
        Khởi tạo và dựng cây KD-Tree từ tập bản ghi đầu vào.
        
        Tham số:
            records: Danh sách VectorRecord.
        """
        self.root = self._build_recursive(records, depth=0)

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm K láng giềng gần nhất trên cây KD-Tree có áp dụng kỹ thuật tỉa nhánh (Pruning).
        
        Các bước thực hiện gợi ý:
            1. Dùng một Bounded Max-Heap (hoặc danh sách ưu tiên) để lưu trữ Top-K láng giềng gần nhất.
            2. Viết hàm đệ quy phụ `_search_node(node)`:
               a. Nếu node is None: kết thúc nhánh.
               b. Tính khoảng cách hình học Euclidean giữa `query_values` và `node.record.values`.
               c. Đẩy (khoảng cách, node.record) vào Heap Top-K.
               d. Xác định trục axis = node.axis.
               e. So sánh tọa độ: diff = query_values[axis] - node.record.values[axis].
               f. Xác định nhánh ưu tiên (near_node) và nhánh đối diện (far_node):
                  - Nếu diff < 0: near_node = node.left, far_node = node.right.
                  - Ngược lại: near_node = node.right, far_node = node.left.
               g. Đệ quy duyệt nhánh ưu tiên trước: `_search_node(near_node)`.
               h. KIỂM TRA ĐIỀU KIỆN TỈA NHÁNH (Pruning):
                  - Khoảng cách từ query tới siêu phẳng phân cách là |diff|.
                  - Nếu heap chưa đầy Top-K HOẶC |diff| < khoảng cách tới phần tử xa nhất trong Heap:
                    Bắt buộc phải duyệt thêm nhánh đối diện: `_search_node(far_node)`.
            3. Bắt đầu đệ quy từ `self.root`.
            4. Trích xuất và sắp xếp kết quả từ Heap trả về danh sách Top-K.
        """
        raise NotImplementedError("Cài đặt thuật toán tìm kiếm KNN trên KD-Tree.")
