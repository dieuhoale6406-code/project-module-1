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
        seed: int = 42,
        metric: str = "cosine",
    ) -> None:
        super().__init__(name="LSH (Random Hyperplane)")
        if dimension <= 0:
            raise ValueError(f"dimension phải là số nguyên dương, nhận: {dimension}")
        if num_hyperplanes <= 0:
            raise ValueError(f"num_hyperplanes phải là số nguyên dương, nhận: {num_hyperplanes}")

        norm_metric = metric.lower().strip()
        if norm_metric not in ("cosine", "cosine_distance", "euclidean", "euclidean_distance"):
            raise ValueError(
                f"LSHIndex chỉ hỗ trợ metric 'cosine' hoặc 'euclidean', nhận: {metric}"
            )

        self.dimension = dimension
        self.num_hyperplanes = num_hyperplanes
        self.seed = seed
        self.metric = norm_metric
        # Danh sách m vector pháp tuyến ngẫu nhiên đại diện cho m siêu phẳng
        self.hyperplanes: List[List[float]] = []
        # Bảng băm: chuỗi_bit_hash -> Danh sách các VectorRecord
        self.buckets: Dict[str, List[Any]] = {}

    def _generate_hyperplanes(self) -> None:
        """
        Sinh m vector ngẫu nhiên từ phân phối chuẩn làm siêu phẳng phân cách.
        
        Các bước thực hiện gợi ý:
            1. Khởi tạo bộ sinh số ngẫu nhiên: random.Random(self.seed).
            2. Tạo `self.num_hyperplanes` vector, mỗi vector có đúng `self.dimension` phần tử.
            3. Mỗi phần tử được lấy từ phân phối chuẩn Gauss: random.gauss(0.0, 1.0).
            4. Lưu danh sách vào `self.hyperplanes`.
        """
        rng = random.Random(self.seed)

        self.hyperplanes = []

        for _ in range(self.num_hyperplanes):
            hyperplane = []

            for _ in range(self.dimension):
                hyperplane.append(rng.gauss(0.0, 1.0))
                
            self.hyperplanes.append(hyperplane)

    def _hash_vector(self, values: List[float]) -> str:
        """
        Ánh xạ một vector d chiều thành một chuỗi bit độ dài m (ví dụ: '10110010').
        
        Các bước thực hiện gợi ý:
            1. Duyệt qua từng siêu phẳng `hyperplane` trong `self.hyperplanes`.
            2. Tính tích vô hướng giữa vector `values` và `hyperplane`:
               dot_prod = sum(v * p for v, p in zip(values, hyperplane))
            3. Nếu dot_prod >= 0: gán bit '1', ngược lại gán bit '0'.
            4. Nối các bit lại thành chuỗi chuỗi ký tự string và trả về.
        """
        if len(values) != self.dimension:
            raise ValueError("Vector không đúng số chiều của chỉ mục.")

        bits = []

        for hyperplane in self.hyperplanes:
            dot_prod = sum(v * p for v,p in zip(values, hyperplane))

            if dot_prod >= 0:
                bits.append("1")
            else:
                bits.append("0")

        return "".join(bits)

    def build(self, records: List[Any]) -> None:
        """
        Xây dựng chỉ mục (Index) băm LSH.
        
        Các bước thực hiện gợi ý:
            1. Gọi `self._generate_hyperplanes()` để chuẩn bị các siêu phẳng.
            2. Khởi tạo lại từ điển rỗng: `self.buckets = {}`.
            3. Duyệt qua từng bản ghi `record` trong `records`:
               a. Tính mã băm: hash_key = self._hash_vector(record.values).
               b. Nếu hash_key chưa có trong buckets, tạo danh sách rỗng: self.buckets[hash_key] = [].
               c. Thêm `record` vào danh sách của bucket tương ứng: self.buckets[hash_key].append(record).
        """
        # Kiểm tra dữ liệu trước khi thay đổi chỉ mục.
        for record in records:
            if len(record.values) != self.dimension:
                raise ValueError(
                    f"Mỗi vector phải có {self.dimension} chiều."
                )

        self._generate_hyperplanes()
        self.buckets = {}

        for record in records:
            hash_key = self._hash_vector(record.values)

            if hash_key not in self.buckets:
                self.buckets[hash_key] = []

            self.buckets[hash_key].append(record)

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector tương đồng bằng băm LSH.
        
        Các bước thực hiện gợi ý:
            1. Băm vector truy vấn để lấy bucket đích: query_hash = self._hash_vector(query_values).
            2. Thu thập danh sách ứng viên (candidate_records):
               - Lấy toàn bộ vector trong bucket `self.buckets.get(query_hash, [])`.
               - Mở rộng (Fallback): Nếu bucket hiện tại có ít hơn `top_k` vector, duyệt thêm
                 các bucket lân cận có khoảng cách Hamming = 1 (chỉ khác đúng 1 bit).
               - Nếu vẫn không có ứng viên: lấy ngẫu nhiên một số vector để không trả về rỗng.
            3. Tính khoảng cách/độ tương đồng Cosine chính xác giữa `query_values` và từng ứng viên.
            4. Sắp xếp danh sách ứng viên theo điểm số và trả về đúng `top_k` kết quả tốt nhất.
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

        if len(self.hyperplanes) != self.num_hyperplanes:
            raise RuntimeError("Cần gọi build() trước khi search().")

        if not self.buckets:
            return []

        # 1. Lấy ứng viên trong bucket trùng mã.
        query_hash = self._hash_vector(query_values)

        candidate_records = list(
            self.buckets.get(query_hash, [])
        )

        # 2. Nếu thiếu, lấy thêm các bucket khác đúng một bit.
        if len(candidate_records) < top_k:
            for i in range(len(query_hash)):
                flipped_bit = (
                    "0" if query_hash[i] == "1" else "1"
                )

                neighbor_hash = (
                    query_hash[:i]
                    + flipped_bit
                    + query_hash[i + 1:]
                )

                candidate_records.extend(
                    self.buckets.get(neighbor_hash, [])
                )

        # 3. Nếu vẫn rỗng, lấy mẫu ngẫu nhiên.
        if not candidate_records:
            all_records = []

            for bucket in self.buckets.values():
                all_records.extend(bucket)

            rng = random.Random(self.seed)
            sample_size = min(top_k, len(all_records))

            candidate_records = rng.sample(
                all_records,
                k=sample_size,
            )

        # 4. Chấm điểm chính xác trên tập ứng viên theo độ đo cấu hình.
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

        # 5. Khoảng cách càng nhỏ càng tốt.
        results.sort(key=lambda item: item[1])

        return results[:top_k]
