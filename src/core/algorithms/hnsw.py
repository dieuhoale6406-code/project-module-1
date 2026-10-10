import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from .base import BaseVectorIndex


class HNSWNode:
    """
    Đỉnh trên đồ thị HNSW đa tầng.
    
    Thuộc tính:
        record: Đối tượng VectorRecord chứa dữ liệu gốc.
        level: Tầng cao nhất mà đỉnh này xuất hiện (từ 0 đến max_level).
        neighbors: Từ điển lưu danh sách kề theo từng tầng:
                   level_idx -> List[HNSWNode] (các đỉnh láng giềng tại tầng đó).
    """

    def __init__(self, record: Any, level: int) -> None:
        self.record = record
        self.level = level
        self.neighbors: Dict[int, List["HNSWNode"]] = {l: [] for l in range(level + 1)}


class HNSWIndex(BaseVectorIndex):
    """
    Thuật toán 5: Hierarchical Navigable Small World (Malkov & Yashunin, 2018)
    Kỹ thuật: Đồ thị thế giới nhỏ phân tầng (Multi-layer Proximity Graph).
    Độ phức tạp: Build O(N log N * d), Search O(log N * d).
    """

    def __init__(
        self,
        dimension: int,
        m: int = 8,              # Số cạnh tối đa mỗi node ở các tầng > 0
        m0: int = 16,            # Số cạnh tối đa mỗi node ở tầng đáy (Level 0)
        ef_construction: int = 32, # Kích thước danh sách ứng viên khi dựng đồ thị
        ef_search: int = 16       # Kích thước danh sách ứng viên khi tìm kiếm
    ) -> None:
        super().__init__(name="HNSW")
        self.dimension = dimension
        self.m = m
        self.m0 = m0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.entry_point: Optional[HNSWNode] = None
        self.max_level: int = -1
        self.nodes: List[HNSWNode] = []
        # Hệ số tỷ lệ phân tầng mL = 1 / ln(M)
        self.ml = 1.0 / math.log(m) if m > 1 else 1.0

    def _get_random_level(self) -> int:
        """
        Sinh tầng ngẫu nhiên theo phân phối xác suất suy giảm hàm mũ (tương tự Skip List).
        
        Công thức: floor(-ln(uniform(0, 1)) * mL).
        Tầng 0 luôn có xác suất 100%, tầng càng cao xác suất xuất hiện càng giảm mạnh.
        """
        r = random.random()
        while r == 0.0:
            r = random.random()
        return int(-math.log(r) * self.ml)

    def _search_layer(
        self, query_values: List[float], enter_node: HNSWNode, ef: int, level: int
    ) -> List[Tuple[HNSWNode, float]]:
        """
        Duyệt tìm kiếm tham lam (Greedy / Beam Search) trên một tầng cụ thể của đồ thị.
        
        Các bước thực hiện gợi ý:
            1. Khởi tạo:
               - v_visited: Tập set() lưu các node đã duyệt qua để tránh lặp.
               - v_candidates: Hàng đợi ưu tiên (Min-Heap) lưu các node đang xét.
               - v_w: Hàng đợi ưu tiên (Max-Heap) lưu `ef` phần tử gần `query_values` nhất tìm được.
            2. Thêm `enter_node` vào cả 3 tập trên.
            3. Vòng lặp: Khi `v_candidates` chưa rỗng:
               a. Lấy node `curr` có khoảng cách nhỏ nhất ra khỏi `v_candidates`.
               b. Lấy phần tử xa nhất trong `v_w` làm mốc so sánh (furthest_in_w).
               c. Nếu khoảng cách của `curr` lớn hơn `furthest_in_w`: Dừng vòng lặp (hội tụ cực tiểu).
               d. Duyệt qua từng đỉnh kề `neighbor` của `curr` tại tầng `level`:
                  - Nếu `neighbor` chưa nằm trong `v_visited`:
                    + Đánh dấu đã thăm.
                    + Tính khoảng cách từ `neighbor` tới `query_values`.
                    + Nếu khoảng cách nhỏ hơn `furthest_in_w` hoặc len(v_w) < ef:
                      * Đẩy `neighbor` vào `v_candidates` và `v_w`.
                      * Nếu len(v_w) > ef: Loại bỏ phần tử xa nhất khỏi `v_w`.
            4. Trả về danh sách các cặp (node, distance) trong `v_w`.
        """
        raise NotImplementedError("Cài đặt duyệt tìm kiếm trên một tầng của HNSW.")

    def build(self, records: List[Any]) -> None:
        """
        Xây dựng đồ thị đa tầng HNSW bằng cách chèn tuần tự từng vector.
        
        Các bước thực hiện gợi ý:
            1. Với bản ghi đầu tiên: Tạo HNSWNode, gán làm `self.entry_point`, max_level = node.level.
            2. Với mỗi bản ghi tiếp theo:
               a. Sinh ngẫu nhiên tầng cho node mới: node_level = self._get_random_level().
               b. Đi từ tầng `self.max_level` xuống tầng `node_level + 1`:
                  - Dùng tìm kiếm tham lam với ef=1 để tìm đỉnh gần nhất làm điểm vào cho tầng kế tiếp.
               c. Từ tầng min(self.max_level, node_level) xuống tầng 0:
                  - Dùng `_search_layer()` với ef=self.ef_construction để tìm tập láng giềng ứng viên.
                  - Chọn ra M láng giềng tốt nhất (ở level 0 chọn M0 láng giềng).
                  - Thêm cạnh hai chiều giữa node mới và các láng giềng này.
                  - Nếu số cạnh của láng giềng vượt quá M (hoặc M0): Tỉa bớt cạnh xa nhất.
               d. Nếu node_level > self.max_level:
                  - Cập nhật self.entry_point = node mới, self.max_level = node_level.
        """
        raise NotImplementedError("Cài đặt thuật toán dựng đồ thị HNSW.")

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector tương đồng trên đồ thị HNSW.
        
        Các bước thực hiện gợi ý:
            1. Nếu đồ thị rỗng (`self.entry_point is None`), trả về [].
            2. Bắt đầu từ `curr = self.entry_point` tại tầng `self.max_level`.
            3. Dẫn đường nhanh ở các tầng cao (từ self.max_level xuống tầng 1):
               - Ở mỗi tầng, tìm đỉnh láng giềng gần `query_values` nhất (ef=1).
               - Đỉnh đó trở thành điểm vào cho tầng tiếp theo ngay bên dưới.
            4. Tại tầng đáy (Level 0):
               - Chạy `_search_layer()` với kích thước chùm tìm kiếm `ef = max(self.ef_search, top_k)`.
               - Sắp xếp kết quả theo khoảng cách tăng dần.
               - Lấy đúng `top_k` phần tử đầu tiên và trả về danh sách (node.record, distance).
        """
        raise NotImplementedError("Cài đặt thuật toán tìm kiếm trên đồ thị HNSW.")
