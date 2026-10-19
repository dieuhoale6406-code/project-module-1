import concurrent.futures
import math
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


def get_optimal_worker_count(cpu_count: Optional[int] = None) -> int:
    """
    Xác định số lượng worker tiến trình tối ưu dựa trên cấu hình phần cứng.
    Nguyên tắc:
    - 0 hoặc 1 core: 1 worker
    - 2 core: 1 worker (để lại 1 core cho OS/Streamlit)
    - 3 - 4 core: cpu_count - 1
    - > 4 core: min(cpu_count - 2, 8) (giới hạn an toàn, tránh nghẽn I/O và ngốn RAM)
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    if cpu_count <= 4:
        return cpu_count - 1
    return min(cpu_count - 2, 8)


def _parse_lines_chunk(lines: List[str], expected_dim: int) -> List[Tuple[str, List[float]]]:
    """
    Hàm worker độc lập cấp module (top-level) dùng cho đa tiến trình.
    Parse một khối dòng text thô thành danh sách (label, values).
    """
    results: List[Tuple[str, List[float]]] = []
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        tokens = raw.split()
        if len(tokens) <= expected_dim:
            continue

        label = " ".join(tokens[:-expected_dim]).strip().lower()
        if not label:
            continue

        try:
            values = [float(x) for x in tokens[-expected_dim:]]
        except ValueError:
            continue

        if not all(math.isfinite(v) for v in values):
            continue

        results.append((label, values))
    return results


def _iter_chunks(file_path: Path, chunk_size: int, limit: Optional[int]) -> Iterator[List[str]]:
    """Tách file thành từng batch dòng để nạp cho các worker."""
    current_chunk: List[str] = []
    lines_read = 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            current_chunk.append(line)
            lines_read += 1

            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []

            if limit is not None and lines_read >= limit:
                break

    if current_chunk:
        yield current_chunk


class TextVectorStorage:
    """
    Tầng lưu trữ dữ liệu Text/CSV theo chuẩn định dạng Stanford GloVe.
    Nhiệm vụ: Đọc định dạng text GloVe 'word v1 v2 ... vd' và ghi thêm bản ghi mới.
    Hỗ trợ chế độ đa tiến trình (Multiprocessing) tự động thích ứng với cấu hình CPU máy tính.
    """

    def __init__(self, file_path: str = "data/vectors.txt", expected_dim: int = 50) -> None:
        self.file_path = Path(file_path)
        self.expected_dim = expected_dim

    def load_vectors(
        self,
        limit: Optional[int] = None,
        force_multiprocessing: bool = False,
        chunk_size: int = 10000,
    ) -> List[Tuple[str, List[float]]]:
        """
        Đọc file vectors.txt và trả về danh sách (label, values).
        Tự động chọn đơn luồng (nếu dữ liệu nhỏ) hoặc đa tiến trình thích ứng phần cứng (nếu dữ liệu lớn).
        """
        if not self.file_path.is_file():
            return []

        # Ngưỡng kích hoạt đa tiến trình:
        # Nếu limit <= 5000 hoặc chỉ có 1 core CPU, chạy đơn luồng in-process để tránh overhead
        num_cores = os.cpu_count() or 1
        use_multiprocessing = force_multiprocessing or (
            (limit is None or limit > 5000) and num_cores > 1
        )

        if not use_multiprocessing:
            return self._load_single_thread(limit=limit)

        workers = get_optimal_worker_count(num_cores)
        results: List[Tuple[str, List[float]]] = []

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                chunks = list(_iter_chunks(self.file_path, chunk_size=chunk_size, limit=limit))
                futures = [
                    executor.submit(_parse_lines_chunk, chunk, self.expected_dim)
                    for chunk in chunks
                ]
                for fut in futures:
                    results.extend(fut.result())
                    if limit is not None and len(results) >= limit:
                        results = results[:limit]
                        break
            return results
        except Exception:
            # Dự phòng an toàn: nếu môi trường không cho phép spawn process, fallback sang đơn luồng
            return self._load_single_thread(limit=limit)

    def _load_single_thread(self, limit: Optional[int] = None) -> List[Tuple[str, List[float]]]:
        results: List[Tuple[str, List[float]]] = []
        with self.file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if limit is not None and len(results) >= limit:
                    break
                parsed = _parse_lines_chunk([line], self.expected_dim)
                if parsed:
                    results.extend(parsed)
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
