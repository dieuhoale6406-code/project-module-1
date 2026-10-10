import time
from typing import Any, List, Tuple


class LinearScanSearcher:
    """
    Khung sườn thuật toán Linear Scan cho Thành viên 4 (CNTT 2).
    Nhiệm vụ: Duyệt tuần tự qua toàn bộ N vector, đo thời gian thực thi (ms) làm baseline.
    """

    def __init__(self, records: List[Any]) -> None:
        self.records = records

    def search_top_k(
        self, query_point: List[float], k: int, metric: str = "cosine"
    ) -> Tuple[List[Tuple[Any, float]], float]:
        """
        Thực hiện quét tuyến tính:
        1. Đo thời gian bắt đầu bằng time.perf_counter().
        2. Tính khoảng cách/độ đo với từng bản ghi trong self.records.
        3. Thu thập Top-K kết quả tối ưu.
        4. Trả về (kết quả Top-K, thời gian thực thi ms).
        """
        raise NotImplementedError("Cài đặt thuật toán quét tuần tự Linear Scan.")
