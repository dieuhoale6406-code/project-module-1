import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..metrics import euclidean_distance
from ..structures.heap import BoundedMaxHeap
from .base import BaseVectorIndex


class IVFIndex(BaseVectorIndex):
    """
    Thuật toán 4: Inverted File Index - Flat (Sivic & Zisserman, 2003 / FAISS)
    Kỹ thuật: Phân cụm K-Means tạo ô Voronoi và chỉ mục đảo (Inverted Lists).
    Độ phức tạp: Build O(I * N * C * d), Search O(C * d + nprobe * (N/C) * d).
    """

    def __init__(
        self,
        dimension: int,
        num_clusters: int = 8,
        nprobe: int = 2,
        seed: Optional[int] = 42,
        metric: str = "euclidean",
    ) -> None:
        super().__init__(name="IVF-Flat")
        if dimension <= 0:
            raise ValueError(f"dimension phải là số nguyên dương, nhận: {dimension}")
        if num_clusters <= 0:
            raise ValueError(f"num_clusters phải là số nguyên dương, nhận: {num_clusters}")
        if nprobe <= 0:
            raise ValueError(f"nprobe phải là số nguyên dương, nhận: {nprobe}")

        self.dimension = dimension
        self.num_clusters = num_clusters
        self.nprobe = nprobe
        self.seed = seed

        norm_metric = metric.lower().strip()
        if norm_metric not in ("euclidean", "euclidean_distance"):
            raise ValueError(f"IVFIndex chỉ hỗ trợ metric 'euclidean', nhận: {metric}")
        self.metric = norm_metric
        # Danh sách tọa độ tâm cụm (centroids): List[List[float]]
        self.centroids: List[List[float]] = []
        # Danh sách chỉ mục đảo: cluster_idx -> List[VectorRecord]
        self.inverted_lists: Dict[int, List[Any]] = {}

    @staticmethod
    def _squared_distance(v1: List[float], v2: List[float]) -> float:
        """Tính bình phương khoảng cách Euclidean giữa hai vector để tối ưu so sánh."""
        return sum((x - y) ** 2 for x, y in zip(v1, v2))

    def _train_kmeans(self, records: List[Any], max_iter: int = 15) -> None:
        """
        Cài đặt thuật toán K-Means thuần túy (không dùng numpy/scikit-learn).
        """
        if not records:
            self.centroids = []
            return

        num_records = len(records)
        # Nếu số bản ghi ít hơn hoặc bằng số cụm mong muốn, lấy tất cả làm tâm cụm
        if num_records <= self.num_clusters:
            self.centroids = [list(rec.values) for rec in records]
            return

        rng = random.Random(self.seed) if self.seed is not None else random.Random()

        # 1. Khởi tạo K tâm cụm ban đầu: Chọn ngẫu nhiên `num_clusters` vector từ `records`
        initial_records = rng.sample(records, self.num_clusters)
        self.centroids = [list(rec.values) for rec in initial_records]

        dim = self.dimension

        # 2. Vòng lặp tối ưu hóa K-Means
        for _ in range(max_iter):
            # a. Tạo K cụm rỗng
            clusters: List[List[Any]] = [[] for _ in range(self.num_clusters)]

            # b. Gán từng vector vào tâm cụm gần nhất
            for rec in records:
                rec_vals = rec.values
                best_c = 0
                min_sq_dist = float("inf")
                for c_idx, centroid in enumerate(self.centroids):
                    sq_dist = self._squared_distance(rec_vals, centroid)
                    if sq_dist < min_sq_dist:
                        min_sq_dist = sq_dist
                        best_c = c_idx
                clusters[best_c].append(rec)

            # c. Cập nhật lại tọa độ các tâm cụm (Centroid Update)
            new_centroids: List[List[float]] = []
            total_shift = 0.0

            for c_idx in range(self.num_clusters):
                cluster = clusters[c_idx]
                c_size = len(cluster)
                if c_size > 0:
                    # Tính trung bình cộng tọa độ các vector trong cụm
                    updated_c = [
                        sum(rec.values[d] for rec in cluster) / c_size
                        for d in range(dim)
                    ]
                else:
                    # Cụm rỗng: gán lại ngẫu nhiên từ một vector trong records
                    random_rec = rng.choice(records)
                    updated_c = list(random_rec.values)

                # Đo độ dịch chuyển của tâm cụm này
                shift = math.sqrt(self._squared_distance(self.centroids[c_idx], updated_c))
                total_shift += shift
                new_centroids.append(updated_c)

            self.centroids = new_centroids

            # d. Kiểm tra hội tụ: nếu tổng độ dịch chuyển cực nhỏ (< 1e-4) thì dừng sớm
            if total_shift < 1e-4:
                break

    def build(self, records: List[Any]) -> None:
        """
        Xây dựng bảng chỉ mục đảo (Inverted Lists).
        """
        if not records:
            self.centroids = []
            self.inverted_lists = {}
            return

        # 1. Kiểm tra tính hợp lệ và đồng nhất về số chiều của các vector
        for idx, rec in enumerate(records):
            rec_vals = getattr(rec, "values", None)
            if rec_vals is None or len(rec_vals) != self.dimension:
                actual_dim = len(rec_vals) if rec_vals is not None else 0
                raise ValueError(
                    f"Vector tại vị trí {idx} có số chiều {actual_dim}, "
                    f"không khớp với số chiều của chỉ mục ({self.dimension})."
                )

        # 2. Huấn luyện phân cụm K-Means để xác định các tâm cụm
        self._train_kmeans(records)

        # 3. Khởi tạo danh sách chỉ mục đảo rỗng cho từng cụm
        num_centroids = len(self.centroids)
        self.inverted_lists = {c_idx: [] for c_idx in range(num_centroids)}

        # 4. Phân phối từng bản ghi vào danh sách chỉ mục đảo của cụm gần nhất
        for rec in records:
            rec_vals = rec.values
            best_c = 0
            min_sq_dist = float("inf")
            for c_idx, centroid in enumerate(self.centroids):
                sq_dist = self._squared_distance(rec_vals, centroid)
                if sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
                    best_c = c_idx
            self.inverted_lists[best_c].append(rec)

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector thông qua cơ chế duyệt nprobe cụm gần nhất.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k phải là số nguyên dương (> 0), nhận được: {top_k}.")

        if not query_values:
            raise ValueError("query_values không được rỗng.")

        for val in query_values:
            if not math.isfinite(val):
                raise ValueError("query_values chứa giá trị không hợp lệ (NaN hoặc Infinity).")

        if len(query_values) != self.dimension:
            raise ValueError(
                f"Số chiều của query ({len(query_values)}) không khớp với số chiều của chỉ mục ({self.dimension})."
            )

        if not self.centroids or not self.inverted_lists:
            return []

        # 1. Tính khoảng cách tới tất cả các centroid và chọn ra nprobe cụm gần nhất
        centroid_dists = [
            (self._squared_distance(query_values, c), c_idx)
            for c_idx, c in enumerate(self.centroids)
        ]
        centroid_dists.sort(key=lambda item: item[0])

        eff_nprobe = min(self.nprobe, len(self.centroids))
        selected_clusters = [c_idx for _, c_idx in centroid_dists[:eff_nprobe]]

        # 2. Gom toàn bộ vector ứng viên từ nprobe cụm này
        candidates: List[Any] = []
        for c_idx in selected_clusters:
            candidates.extend(self.inverted_lists.get(c_idx, []))

        if not candidates:
            return []

        # 3. Sử dụng BoundedMaxHeap để lọc ra đúng Top-K kết quả gần nhất
        heap = BoundedMaxHeap(capacity=top_k)
        for record in candidates:
            dist = euclidean_distance(query_values, record.values)
            heap.push(dist, record)

        return heap.get_sorted_results()
