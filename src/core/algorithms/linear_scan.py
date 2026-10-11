import math
from typing import Any, List, Optional, Tuple

from ..metrics import cosine_similarity, euclidean_distance
from .base import BaseVectorIndex


class LinearScanIndex(BaseVectorIndex):
    """
    Thuật toán 1: Exact Linear Scan (Baseline)
    Độ phức tạp: Build O(1), Search O(N * d).
    Vai trò: Chuẩn đối chiếu Ground Truth tuyệt đối (Recall = 100%).
    Cơ chế: Brute Force duyệt toàn bộ N vector và Full Sorting theo (distance, record.id).
    """

    def __init__(self, metric: str = "cosine") -> None:
        super().__init__(name="Linear Scan")
        norm_metric = metric.lower().strip()
        if norm_metric not in ("cosine", "cosine_distance", "euclidean", "euclidean_distance"):
            raise ValueError(
                f"Metric '{metric}' không được hỗ trợ. Chỉ hỗ trợ 'cosine' hoặc 'euclidean'."
            )
        self.metric = norm_metric
        self.records: List[Any] = []
        self.dimension: Optional[int] = None

    def build(self, records: List[Any]) -> None:
        """
        Nạp toàn bộ danh sách VectorRecord vào bộ nhớ RAM và kiểm tra tính nhất quán về số chiều.

        Tham số:
            records: Danh sách các đối tượng VectorRecord chứa label và tọa độ vector.
        """
        if not records:
            self.records = []
            self.dimension = None
            return

        # Xác định số chiều từ bản ghi đầu tiên
        first_dim = getattr(records[0], "dimension", len(records[0].values))
        for idx, rec in enumerate(records):
            rec_dim = getattr(rec, "dimension", len(rec.values))
            if rec_dim != first_dim:
                raise ValueError(
                    f"Vector tại vị trí {idx} có số chiều {rec_dim}, không khớp với số chiều chuẩn {first_dim}."
                )

        self.records = list(records)
        self.dimension = first_dim

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Duyệt tuần tự qua toàn bộ N vector để tìm Top-K vector gần nhất (Exact KNN).

        Các bước thực hiện:
            1. Kiểm tra tính hợp lệ của query_values và top_k.
            2. Nếu chỉ mục rỗng, trả về danh sách rỗng.
            3. Duyệt toàn bộ records, tính distance tương ứng (Euclidean hoặc Cosine Distance).
            4. Sắp xếp tăng dần theo cặp khóa (distance, record.id) để đảm bảo tính ổn định (tie-breaking).
            5. Trả về tối đa Top-K kết quả: List[(VectorRecord, distance)].

        Tham số:
            query_values: Mảng số thực tọa độ của vector truy vấn.
            top_k: Số lượng kết quả gần nhất cần lấy.

        Trả về:
            Danh sách gồm các tuple (VectorRecord, khoảng_cách).
        """
        if top_k <= 0:
            raise ValueError(f"top_k phải là số nguyên dương (> 0), nhận được: {top_k}.")
        if not query_values:
            raise ValueError("query_values không được rỗng.")

        # Kiểm tra NaN và Infinity trong query_values
        for val in query_values:
            if not math.isfinite(val):
                raise ValueError("query_values chứa giá trị không hợp lệ (NaN hoặc Infinity).")

        # Nếu tập dữ liệu rỗng, trả về danh sách rỗng
        if not self.records:
            return []

        # Kiểm tra tính nhất quán số chiều giữa query và tập dữ liệu
        if self.dimension is not None and len(query_values) != self.dimension:
            raise ValueError(
                f"Số chiều của query ({len(query_values)}) không khớp với số chiều của chỉ mục ({self.dimension})."
            )

        # Tính trước norm của query_values nếu dùng Cosine
        query_norm = 0.0
        is_cosine = self.metric in ("cosine", "cosine_distance")
        if is_cosine:
            sq_sum = sum(q * q for q in query_values)
            query_norm = math.sqrt(sq_sum)

        candidates = []
        for record in self.records:
            rec_values = record.values
            if is_cosine:
                rec_norm = getattr(record, "norm", None)
                if rec_norm is None:
                    rec_norm = math.sqrt(sum(v * v for v in rec_values))
                sim = cosine_similarity(query_values, rec_values, query_norm, rec_norm)
                # Cosine distance = 1.0 - Cosine similarity (càng gần thì distance càng nhỏ)
                dist = 1.0 - sim
            else:
                dist = euclidean_distance(query_values, rec_values)

            # Cặp khóa (distance, record.id) để tie-breaking ổn định
            rec_id = str(getattr(record, "id", ""))
            candidates.append((dist, rec_id, record))

        # Sắp xếp tăng dần theo (distance, record.id)
        candidates.sort(key=lambda item: (item[0], item[1]))

        # Lấy tối đa Top-K kết quả
        top_candidates = candidates[:top_k]
        return [(item[2], item[0]) for item in top_candidates]

