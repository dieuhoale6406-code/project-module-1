import math
from typing import Any, Dict, List
from .base import BaseEntity


class VectorRecord(BaseEntity):
    """
    Thực thể biểu diễn một từ vựng và vector số thực đa chiều trong không gian nhúng.
    Đóng vai trò là vật chứa dữ liệu thuần túy (Data Holder), không chứa thuật toán tìm kiếm.
    """

    def __init__(self, record_id: str, label: str, values: List[float]) -> None:
        super().__init__(record_id)
        self._label = str(label).strip().lower()
        self._values = [float(v) for v in values]
        # Tính trước chuẩn Euclidean L2 (Norm) để các thuật toán tối ưu phép chia
        sq_sum = sum(v * v for v in self._values)
        self._norm = math.sqrt(sq_sum) if sq_sum > 0.0 else 1e-9

    @property
    def label(self) -> str:
        """Nhãn từ vựng (ví dụ: 'king', 'queen', 'vietnam')."""
        return self._label

    @property
    def values(self) -> List[float]:
        """Danh sách tọa độ số thực của vector."""
        return self._values

    @property
    def dimension(self) -> int:
        """Số chiều của không gian vector (d)."""
        return len(self._values)

    @property
    def norm(self) -> float:
        """Độ dài chuẩn L2 của vector: sqrt(sum(v_i^2))."""
        return self._norm

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thực thể thành dictionary nguyên bản."""
        return {
            "id": self._id,
            "label": self._label,
            "values": self._values,
            "dimension": self.dimension,
            "norm": self._norm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        """Khởi tạo thực thể từ dictionary."""
        return cls(
            record_id=data["id"],
            label=data["label"],
            values=data["values"],
        )

    @classmethod
    def from_line(cls, line_idx: int, line_str: str) -> "VectorRecord":
        """
        Khởi tạo thực thể trực tiếp từ một dòng file text chuẩn Stanford GloVe:
        Cấu trúc: 'word v1 v2 v3 ... vd'
        """
        parts = line_str.strip().split()
        if not parts:
            raise ValueError("Dòng dữ liệu rỗng.")
        label = parts[0]
        values = [float(x) for x in parts[1:]]
        return cls(record_id=f"VEC_{line_idx:05d}", label=label, values=values)

    def __repr__(self) -> str:
        preview = ", ".join(f"{v:.2f}" for v in self._values[:3])
        if self.dimension > 3:
            preview += ", ..."
        return f"<VectorRecord id='{self._id}' label='{self._label}' dim={self.dimension} values=[{preview}]>"
