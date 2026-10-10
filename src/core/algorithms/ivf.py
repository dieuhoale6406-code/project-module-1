import random
from typing import Any, Dict, List, Tuple
from .base import BaseVectorIndex


class IVFIndex(BaseVectorIndex):
    """
    Thuật toán 4: Inverted File Index - Flat (Sivic & Zisserman, 2003 / FAISS)
    Kỹ thuật: Phân cụm K-Means tạo ô Voronoi và chỉ mục đảo (Inverted Lists).
    Độ phức tạp: Build O(I * N * C * d), Search O(C * d + nprobe * (N/C) * d).
    """

    def __init__(self, dimension: int, num_clusters: int = 8, nprobe: int = 2) -> None:
        super().__init__(name="IVF-Flat")
        self.dimension = dimension
        self.num_clusters = num_clusters
        self.nprobe = nprobe
        # Danh sách tọa độ tâm cụm (centroids): List[List[float]]
        self.centroids: List[List[float]] = []
        # Danh sách chỉ mục đảo: cluster_idx -> List[VectorRecord]
        self.inverted_lists: Dict[int, List[Any]] = {}

    def _train_kmeans(self, records: List[Any], max_iter: int = 15) -> None:
        """
        Cài đặt thuật toán K-Means thuần túy (không dùng numpy/scikit-learn).
        
        Các bước thực hiện gợi ý:
            1. Khởi tạo K tâm cụm ban đầu: Chọn ngẫu nhiên `self.num_clusters` vector từ `records`.
            2. Vòng lặp tối ưu (lặp từ 1 đến `max_iter`):
               a. Tạo K cụm rỗng: clusters = [[] for _ in range(self.num_clusters)].
               b. Gán từng vector vào tâm cụm gần nhất (dùng khoảng cách Euclidean):
                  - Với mỗi record, tìm centroid có khoảng cách nhỏ nhất, đưa record vào cụm đó.
               c. Cập nhật lại tọa độ các tâm cụm (Centroid Update):
                  - Với mỗi cụm có chứa vector: Tính trung bình cộng tọa độ các vector trong cụm:
                    new_centroid[j] = sum(v[j] for v in cluster) / len(cluster).
                  - Nếu cụm rỗng: giữ nguyên centroid cũ hoặc gán lại ngẫu nhiên.
               d. Kiểm tra hội tụ (nếu tọa độ centroids thay đổi cực nhỏ thì dừng sớm).
            3. Lưu danh sách tâm cụm cuối cùng vào `self.centroids`.
        """
        raise NotImplementedError("Cài đặt thuật toán phân cụm K-Means cho IVF.")

    def build(self, records: List[Any]) -> None:
        """
        Xây dựng bảng chỉ mục đảo (Inverted Lists).
        
        Các bước thực hiện gợi ý:
            1. Gọi `self._train_kmeans(records)` để tìm ra `self.num_clusters` tâm cụm.
            2. Khởi tạo danh sách chỉ mục đảo rỗng cho từng cụm:
               self.inverted_lists = {c_idx: [] for c_idx in range(len(self.centroids))}.
            3. Duyệt qua từng bản ghi `record` trong `records`:
               a. Tìm chỉ số cụm `best_c` có tâm cụm gần `record.values` nhất.
               b. Đưa bản ghi vào danh sách chỉ mục đảo của cụm đó:
                  self.inverted_lists[best_c].append(record).
        """
        raise NotImplementedError("Cài đặt dựng chỉ mục đảo IVF.")

    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector thông qua cơ chế duyệt nprobe cụm gần nhất.
        
        Các bước thực hiện gợi ý:
            1. Tính khoảng cách Euclidean từ `query_values` tới tất cả `self.centroids`.
            2. Sắp xếp các centroid theo khoảng cách tăng dần, chọn ra `self.nprobe` centroid gần nhất.
            3. Gom toàn bộ vector nằm trong danh sách chỉ mục đảo của `nprobe` centroid này:
               candidates = []
               for c_idx in nearest_centroids:
                   candidates.extend(self.inverted_lists.get(c_idx, []))
            4. Quét tuần tự tính khoảng cách chính xác giữa `query_values` và từng vector trong `candidates`.
            5. Sắp xếp danh sách ứng viên và trả về đúng `top_k` kết quả tối ưu nhất.
        """
        raise NotImplementedError("Cài đặt tìm kiếm nprobe cụm trên IVF.")
