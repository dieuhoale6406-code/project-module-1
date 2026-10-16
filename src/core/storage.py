import math
from pathlib import Path
from typing import List, Optional, Tuple


class TextVectorStorage:
    """
    Tầng lưu trữ dữ liệu Text/CSV theo chuẩn định dạng Stanford GloVe.
    Nhiệm vụ: Đọc định dạng text GloVe 'word v1 v2 ... vd' và ghi thêm bản ghi mới.
    """

    def __init__(self, file_path: str = "data/vectors.txt", expected_dim: int = 50) -> None:
        self.file_path = Path(file_path)
        self.expected_dim = expected_dim

    def load_vectors(self, limit: Optional[int] = None) -> List[Tuple[str, List[float]]]:
        """
        Đọc file vectors.txt và trả về danh sách (label, values).
        Tự động tiền xử lý dữ liệu và bỏ qua các dòng lỗi hoặc sai số chiều.
        """
        if not self.file_path.is_file():
            return []

        results: List[Tuple[str, List[float]]] = []

        with self.file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if limit is not None and len(results) >= limit:
                    break

                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue

                tokens = raw.split()
                if len(tokens) <= self.expected_dim:
                    continue

                # Nhãn từ vựng (cho phép nhãn có khoảng trắng)
                label = " ".join(tokens[:-self.expected_dim]).strip().lower()
                if not label:
                    continue

                # 50 giá trị số thực tọa độ
                try:
                    values = [float(x) for x in tokens[-self.expected_dim:]]
                except ValueError:
                    continue

                # Kiểm tra số thực hợp lệ (không chứa NaN hay Infinity)
                if not all(math.isfinite(v) for v in values):
                    continue

                results.append((label, values))

        return results

    def append_vector(self, label: str, values: List[float]) -> bool:
        """
        Ghi thêm một dòng vector mới vào cuối file theo định dạng GloVe.
        """
        clean_label = " ".join(str(label).split()).lower()
        if not clean_label:
            raise ValueError("Nhãn từ vựng không được để trống.")
        if clean_label.startswith("#"):
            raise ValueError("Nhãn từ vựng không được bắt đầu bằng '#'.")

        if not values or len(values) != self.expected_dim:
            actual_dim = len(values) if values else 0
            raise ValueError(
                f"Số chiều vector ({actual_dim}) không khớp với cấu hình hệ thống ({self.expected_dim})."
            )

        for v in values:
            if not math.isfinite(v):
                raise ValueError("Tọa độ vector chứa giá trị không hợp lệ (NaN hoặc Infinity).")

        # Đảm bảo thư mục cha tồn tại
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Định dạng chuẩn: 'label v1 v2 ... vd\n'
        formatted_values = " ".join(f"{v:.6f}" for v in values)
        line = f"{clean_label} {formatted_values}\n"

        needs_newline = False
        if self.file_path.is_file() and self.file_path.stat().st_size > 0:
            with self.file_path.open("rb") as rf:
                rf.seek(-1, 2)
                needs_newline = rf.read(1) != b"\n"

        with self.file_path.open("a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(line)

        return True
