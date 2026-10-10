import time
from typing import Any, List, Tuple
from .base import BaseVectorIndex


class LinearScanIndex(BaseVectorIndex):
    """
    Thuật toán 1: Exact Linear Scan (Baseline)
    Độ phức tạp: Build O(1), Search O(N * d).
    Vai trò: Chuẩn đối chiếu Ground Truth tuyệt đối (Recall = 100%).
    """

    def __init__(self, metric: str = "cosine") -> None:
        super().__init__(name="Linear Scan")
        self.metric = metric
        self.records: List[Any] = []

    def build(self, records: List[Any]) -> None:
        """
        Nạp toàn bộ danh sách VectorRecord vào bộ nhớ RAM.
        
        Tham số:
            records: Danh sách các đối tượng VectorRecord chứa label và tọa độ vector.
        """
        self.records = records

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Duyệt tuần tự qua toàn bộ N vector để tìm Top-K vector gần nhất.
        
        Các bước thực hiện gợi ý:
            1. Bắt đầu bấm giờ đo thời gian bằng time.perf_counter().
            2. Duyệt qua từng bản ghi `record` trong `self.records`.
            3. Tính khoảng cách/độ tương đồng giữa `query_values` và `record.values`
               (gọi hàm tương ứng trong src/core/metrics.py: cosine_similarity hoặc euclidean_distance).
            4. Lưu kết quả vào danh sách các cặp (record, score).
            5. Sắp xếp danh sách (Cosine: giảm dần; Euclidean: tăng dần) hoặc dùng Heap để lấy Top-K.
            6. Trả về đúng `top_k` phần tử tốt nhất.
            
        Tham số:
            query_values: Mảng số thực tọa độ của vector truy vấn.
            top_k: Số lượng kết quả gần nhất cần lấy.
            
        Trả về:
            Danh sách gồm các tuple (VectorRecord, điểm_số).
        """
        raise NotImplementedError("Cài đặt thuật toán duyệt tuần tự Linear Scan.")
