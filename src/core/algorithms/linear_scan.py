import math
from typing import Any, List, Optional, Tuple

from ..metrics import cosine_similarity, euclidean_distance
from ..structures.heap import BoundedMaxHeap
from .base import BaseVectorIndex


class LinearScanIndex(BaseVectorIndex):
    """
    Exact Linear Scan Baseline (Ground Truth).

    Complexity:
        - Build: O(N) memory copy and dimension validation.
        - Search: O(N * D + N log K) distance computations and bounded max-heap maintenance.
    Auxiliary search memory: O(K).
    Role: Absolute ground truth baseline (100% recall).
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
        """Load VectorRecords into memory and validate dimension consistency."""
        if not records:
            self.records = []
            self.dimension = None
            return

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
        """Sequentially scan all vectors to find Top-K nearest neighbors (Exact KNN)."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k phải là số nguyên dương (> 0), nhận được: {top_k}.")

        if not query_values:
            raise ValueError("query_values không được rỗng.")

        for val in query_values:
            if not math.isfinite(val):
                raise ValueError("query_values chứa giá trị không hợp lệ (NaN hoặc Infinity).")

        if not self.records:
            return []

        if self.dimension is not None and len(query_values) != self.dimension:
            raise ValueError(
                f"Số chiều của query ({len(query_values)}) không khớp với số chiều của chỉ mục ({self.dimension})."
            )

        query_norm = 0.0
        is_cosine = self.metric in ("cosine", "cosine_distance")
        if is_cosine:
            sq_sum = sum(q * q for q in query_values)
            query_norm = math.sqrt(sq_sum)
            if query_norm == 0.0:
                raise ValueError(
                    "Không thể tính Cosine Similarity với vector truy vấn có chuẩn độ dài bằng 0 (Zero Vector)."
                )

        heap = BoundedMaxHeap(capacity=top_k)
        for record in self.records:
            rec_values = record.values
            if is_cosine:
                rec_norm = getattr(record, "norm", None)
                if rec_norm is None:
                    rec_norm = math.sqrt(sum(v * v for v in rec_values))
                sim = cosine_similarity(query_values, rec_values, query_norm, rec_norm)
                # Cosine distance = 1.0 - Cosine similarity (smaller is closer).
                dist = 1.0 - sim
            else:
                dist = euclidean_distance(query_values, rec_values)

            heap.push(dist, record)

        return heap.get_sorted_results()
