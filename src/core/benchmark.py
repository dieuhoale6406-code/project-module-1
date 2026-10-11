from typing import Any, Dict, List, Optional
from core.algorithms.base import BaseVectorIndex
from models.record import VectorRecord

class BenchmarkRunner:
    """
    Khung sườn bộ thực thi thử nghiệm hiệu năng cho (Phát - CNTT).
    Nhiệm vụ: Đo lường đa chỉ số và so sánh 5 thuật toán tìm kiếm vector.
    
    Các chỉ số cần đo:
    - Build_Time_ms: Thời gian xây dựng chỉ mục.
    - Mean_Latency_ms: Độ trễ truy vấn trung bình trên tập câu hỏi.
    - P95_Latency_ms: Phân vị thời gian truy vấn 95%.
    - QPS: Số truy vấn giải quyết trong 1 giây (1000 / Mean_Latency).
    - Recall_Percent: Độ chính xác so với Ground Truth của Linear Scan.
    """

    def __init__(self, full_records: List[VectorRecord], dimension: int) -> None:
        self.full_records = full_records
        self.dimension = dimension

    def sample_query_set(self, query_count: int = 50, seed: int = 42) -> List[List[float]]:
        """
        Trích xuất tập truy vấn ngẫu nhiên cố định để thử nghiệm.
        
        Gợi ý:
        Dùng random.seed(seed) để đảm bảo kết quả đo có thể tái lập được.
        """
        raise NotImplementedError("Cài đặt hàm sinh tập truy vấn chuẩn.")

    def run_benchmark(
        self,
        algorithm_instances: Dict[str, BaseVectorIndex],
        scale_sizes: Optional[List[int]] = None,
        top_k: int = 5,
        query_count: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Thực thi ma trận thử nghiệm benchmark trên các mốc quy mô N: [100, 500, 1000, ...].
        
        Các bước thực hiện gợi ý cho:
        1. Sinh tập câu hỏi queries bằng self.sample_query_set().
        2. Với từng mốc quy mô N:
           a. Lấy mẫu N vector: sub_records = self.full_records[:n].
           b. Lấy Ground Truth Top-K từ thuật toán Linear Scan cho từng câu hỏi.
           c. Với từng thuật toán:
              - Bấm giờ đo thời gian build() chỉ mục.
              - Lặp qua tập queries, bấm giờ đo thời gian search() và đếm số hit trùng Ground Truth.
              - Tính toán Mean Latency, P95 Latency, QPS và Recall@K.
        3. Trả về danh sách dictionary chứa kết quả đo đạc để hiển thị lên bảng biểu.
        """
        raise NotImplementedError("Cài đặt quy trình chạy benchmark đa chỉ số.")
