import math
from typing import Any, Dict, List
from .base import BaseEntity


class VectorRecord(BaseEntity):
    """Immutable representation of an embedding vector and its textual label."""

    def __init__(self, record_id: str, label: str, values: List[float]) -> None:
        super().__init__(record_id)
        self._label = str(label).strip().lower()
        parsed_values = []
        for v in values:
            val = float(v)
            if not math.isfinite(val):
                raise ValueError("Tọa độ vector chứa giá trị không hợp lệ (NaN hoặc Infinity).")
            parsed_values.append(val)
        self._values = parsed_values
        # Precompute Euclidean L2 norm to optimize downstream metric computations.
        sq_sum = sum(v * v for v in self._values)
        self._norm = math.sqrt(sq_sum)

    @property
    def label(self) -> str:
        """Textual label (e.g., 'king', 'queen', 'vietnam')."""
        return self._label

    @property
    def values(self) -> List[float]:
        """Vector coordinates (defensive copy)."""
        return list(self._values)

    @property
    def dimension(self) -> int:
        """Dimension of the vector space (d)."""
        return len(self._values)

    @property
    def norm(self) -> float:
        """Precomputed L2 norm of the vector: sqrt(sum(v_i^2))."""
        return self._norm

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity into a dictionary."""
        return {
            "id": self._id,
            "label": self._label,
            "values": self._values,
            "dimension": self.dimension,
            "norm": self._norm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        """Instantiate entity from a dictionary."""
        return cls(
            record_id=data["id"],
            label=data["label"],
            values=data["values"],
        )

    @classmethod
    def from_line(cls, line_idx: int, line_str: str) -> "VectorRecord":
        """Parse a record from a Stanford GloVe format text line: 'word v1 v2 ... vd'."""
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
