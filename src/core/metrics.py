import math
from typing import List


def cosine_similarity(v1: List[float], v2: List[float], norm1: float, norm2: float) -> float:
    """
    Tính độ tương đồng góc Cosine giữa 2 vector.
    Gợi ý: Tận dụng norm1 và norm2 đã tính trước để đạt O(d).
    """
    raise NotImplementedError("Cài đặt hàm cosine_similarity.")


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """
    Tính khoảng cách hình học Euclidean giữa 2 vector.
    """
    raise NotImplementedError("Cài đặt hàm euclidean_distance.")


def dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Tính tích vô hướng giữa 2 vector.
    """
    raise NotImplementedError("Cài đặt hàm dot_product.")
