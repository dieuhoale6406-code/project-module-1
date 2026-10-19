import math
from ..metrics import cosine_similarity, euclidean_distance

import random
from typing import Any, Dict, List, Tuple
from .base import BaseVectorIndex


class LSHIndex(BaseVectorIndex):
    """
    Thuật toán 3: Locality-Sensitive Hashing (Indyk & Motwani, 1998 / Charikar, 2002)
    Kỹ thuật: Random Hyperplane (SimHash) cho góc Cosine Similarity và Euclidean Distance.
    Độ phức tạp: Build O(N * m * d), Search O(m * d + B * d) với B là kích thước bucket.
    """

    def __init__(
        self,
        dimension: int,
        num_hyperplanes: int = 4,
        num_tables: int = 1,
        seed: int = 42,
        metric: str = "cosine",
    ) -> None:
        super().__init__(name="LSH (Random Hyperplane)")
        if dimension <= 0:
            raise ValueError(f"dimension phải là số nguyên dương, nhận: {dimension}")
        if num_hyperplanes <= 0:
            raise ValueError(f"num_hyperplanes phải là số nguyên dương, nhận: {num_hyperplanes}")
        if num_tables <= 0:
            raise ValueError(f"num_tables phải là số nguyên dương, nhận: {num_tables}")

        norm_metric = metric.lower().strip()
        if norm_metric not in ("cosine", "cosine_distance", "euclidean", "euclidean_distance"):
            raise ValueError(
                f"LSHIndex chỉ hỗ trợ metric 'cosine' hoặc 'euclidean', nhận: {metric}"
            )

        self.dimension = dimension
        self.num_hyperplanes = num_hyperplanes
        self.num_tables = num_tables
        self.seed = seed
        self.metric = norm_metric
        # Danh sách m vector pháp tuyến ngẫu nhiên đại diện cho m siêu phẳng (của bảng 0)
        self.hyperplanes: List[List[float]] = []
        # Bảng băm chính (bảng 0): chuỗi_bit_hash -> Danh sách các VectorRecord
        self.buckets: Dict[str, List[Any]] = {}
        # Hỗ trợ đa bảng băm (Multi-Table LSH chuẩn Indyk & Motwani, 1998)
        self.table_hyperplanes: List[List[List[float]]] = []
        self.tables: List[Dict[str, List[Any]]] = []

    def _generate_hyperplanes(self) -> None:
        """
        Sinh các vector ngẫu nhiên từ phân phối chuẩn làm siêu phẳng phân cách cho các bảng băm.
        """
        rng = random.Random(self.seed)
        self.table_hyperplanes = []

        for _ in range(self.num_tables):
            table_planes = []
            for _ in range(self.num_hyperplanes):
                hyperplane = [rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
                table_planes.append(hyperplane)
            self.table_hyperplanes.append(table_planes)

        self.hyperplanes = self.table_hyperplanes[0]

    def _hash_vector(self, values: List[float], table_idx: int = 0) -> str:
        """
        Ánh xạ một vector d chiều thành một chuỗi bit độ dài m trên bảng băm table_idx.
        """
        if len(values) != self.dimension:
            raise ValueError("Vector không đúng số chiều của chỉ mục.")

        planes = (
            self.table_hyperplanes[table_idx]
            if table_idx < len(self.table_hyperplanes)
            else self.hyperplanes
        )
        bits = []

        for hyperplane in planes:
            dot_prod = sum(v * p for v, p in zip(values, hyperplane))
            if dot_prod >= 0:
                bits.append("1")
            else:
                bits.append("0")

        return "".join(bits)

    def build(self, records: List[Any]) -> None:
        """
        Xây dựng chỉ mục (Index) băm LSH đa bảng.
        """
        # Kiểm tra dữ liệu trước khi thay đổi chỉ mục.
        for record in records:
            if len(record.values) != self.dimension:
                raise ValueError(
                    f"Mỗi vector phải có {self.dimension} chiều."
                )

        self._generate_hyperplanes()
        # Đảm bảo đồng bộ nếu self.hyperplanes bị mock/patch từ unit tests
        if not self.table_hyperplanes or self.table_hyperplanes[0] != self.hyperplanes:
            self.table_hyperplanes = [self.hyperplanes]
            self.num_tables = 1

        self.tables = [{} for _ in range(self.num_tables)]
        self.buckets = self.tables[0]

        for record in records:
            for t_idx in range(self.num_tables):
                hash_key = self._hash_vector(record.values, table_idx=t_idx)
                if hash_key not in self.tables[t_idx]:
                    self.tables[t_idx][hash_key] = []
                self.tables[t_idx][hash_key].append(record)

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector tương đồng bằng băm LSH (Multi-Table & Multi-Probe).
        """
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError("top_k phải là số nguyên dương.")

        if len(query_values) != self.dimension:
            raise ValueError(
                f"Vector truy vấn phải có {self.dimension} chiều."
            )

        if not all(math.isfinite(value) for value in query_values):
            raise ValueError("Vector truy vấn phải chứa các số hữu hạn.")

        query_norm = math.sqrt(
            sum(value * value for value in query_values)
        )

        if query_norm == 0:
            raise ValueError("Vector truy vấn không được là vector không.")

        if not self.hyperplanes or len(self.hyperplanes) != self.num_hyperplanes:
            raise RuntimeError("Cần gọi build() trước khi search().")

        if not self.buckets and not any(self.tables):
            return []

        # 1. Thu thập ứng viên trên tất cả các bảng băm kết hợp thăm dò Hamming radius = 1 (Multi-Probe)
        seen_ids = set()
        candidate_records = []

        active_tables = self.tables if self.tables else [self.buckets]
        for t_idx, t_buckets in enumerate(active_tables):
            query_hash = self._hash_vector(query_values, table_idx=t_idx)

            # Bucket chính xác trùng mã
            for record in t_buckets.get(query_hash, []):
                if record.id not in seen_ids:
                    seen_ids.add(record.id)
                    candidate_records.append(record)

            # Multi-probe: Láng giềng khác 1 bit để gom các vector nằm sát vách siêu phẳng
            for i in range(len(query_hash)):
                flipped_bit = "0" if query_hash[i] == "1" else "1"
                neighbor_hash = (
                    query_hash[:i]
                    + flipped_bit
                    + query_hash[i + 1:]
                )
                for record in t_buckets.get(neighbor_hash, []):
                    if record.id not in seen_ids:
                        seen_ids.add(record.id)
                        candidate_records.append(record)

        # 2. Nếu vẫn rỗng, lấy mẫu ngẫu nhiên (Random Fallback)
        if not candidate_records:
            all_records = []
            for t in active_tables:
                for bucket in t.values():
                    all_records.extend(bucket)

            rng = random.Random(self.seed)
            sample_size = min(top_k, len(all_records))
            candidate_records = rng.sample(all_records, k=sample_size)

        # 3. Chấm điểm chính xác trên tập ứng viên theo độ đo cấu hình.
        results = []
        is_cosine = self.metric in ("cosine", "cosine_distance")

        if is_cosine:
            for record in candidate_records:
                similarity = cosine_similarity(
                    query_values,
                    record.values,
                    query_norm,
                    record.norm,
                )
                distance = 1.0 - similarity
                results.append((record, distance))
        else:
            for record in candidate_records:
                dist = euclidean_distance(query_values, record.values)
                results.append((record, dist))

        # 4. Sắp xếp khoảng cách tăng dần và lấy top_k
        results.sort(key=lambda item: item[1])
        return results[:top_k]
