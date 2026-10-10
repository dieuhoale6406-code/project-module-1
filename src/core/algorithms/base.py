from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class BaseVectorIndex(ABC):
    """
    Giao diện trừu tượng chuẩn hóa (Interface) cho tất cả thuật toán tìm kiếm vector.
    Mọi thuật toán đều phải tuân thủ đúng 2 phương thức: build() và search().
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build(self, records: List[Any]) -> None:
        """
        Dựng chỉ mục từ danh sách các bản ghi VectorRecord.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query_values: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm Top-K vector gần nhất.
        Trả về danh sách các tuple: (VectorRecord, khoảng_cách_hoặc_độ_tương_đồng).
        """
        raise NotImplementedError
