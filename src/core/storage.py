from pathlib import Path
from typing import Any, List, Tuple


class TextVectorStorage:
    """
    Khung sườn đọc ghi dữ liệu Text/CSV.
    Nhiệm vụ: Đọc định dạng text GloVe 'word v1 v2 ... vd' và ghi thêm bản ghi mới.
    """

    def __init__(self, file_path: str = "data/vectors.txt") -> None:
        self.file_path = Path(file_path)

    def load_vectors(self) -> List[Tuple[str, List[float]]]:
        """Đọc toàn bộ file vectors.txt và trả về danh sách (label, values)."""
        raise NotImplementedError("Cài đặt phương thức đọc file vectors.txt.")

    def append_vector(self, label: str, values: List[float]) -> bool:
        """Ghi thêm một dòng vector mới vào cuối file."""
        raise NotImplementedError("Cài đặt phương thức ghi thêm vector vào file.")
