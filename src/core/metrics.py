import math
from typing import List


def _validate_vectors(v1: List[float], v2: List[float]) -> None:
    """Validate that vectors are non-empty, equal in length, and contain finite values."""
    if not v1 or not v2:
        raise ValueError("Vector không được rỗng.")
    if len(v1) != len(v2):
        raise ValueError(f"Kích thước hai vector không khớp: {len(v1)} != {len(v2)}.")
    for x, y in zip(v1, v2):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Vector chứa giá trị không hợp lệ (NaN hoặc Infinity).")


def cosine_similarity(v1: List[float], v2: List[float], norm1: float, norm2: float) -> float:
    """Compute Cosine similarity between two vectors using precomputed L2 norms."""
    _validate_vectors(v1, v2)

    if not (math.isfinite(norm1) and math.isfinite(norm2)):
        raise ValueError("Chuẩn độ dài (norm) chứa giá trị không hợp lệ (NaN hoặc Infinity).")
    if norm1 < 0.0 or norm2 < 0.0:
        raise ValueError("Chuẩn độ dài (norm) không được là số âm.")

    # Mathematical convention for zero vectors: avoid division by zero.
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    denom = norm1 * norm2
    if denom == 0.0:
        return 0.0

    dot = sum(x * y for x, y in zip(v1, v2))
    sim = dot / denom

    # Clamp to [-1.0, 1.0] to guard against floating-point inaccuracies.
    if sim > 1.0:
        return 1.0
    elif sim < -1.0:
        return -1.0
    return sim


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    _validate_vectors(v1, v2)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))


def dot_product(v1: List[float], v2: List[float]) -> float:
    """Compute dot product between two vectors."""
    _validate_vectors(v1, v2)
    return sum(x * y for x, y in zip(v1, v2))

