# ĐỒ ÁN MODULE 1: HỆ THỐNG VECTOR SEARCH ENGINE (MINI FAISS / MILVUS)

## 1. Cấu trúc thư mục dự án

```text
├── data/
│   ├── .gitkeep
│   └── vectors.txt                # Dataset từ vựng và vector 50D (Stanford GloVe chuẩn)
├── src/
│   ├── models/                    # Mô hình thực thể dữ liệu
│   │   ├── base.py                # Lớp cơ sở BaseEntity
│   │   ├── record.py              # Thực thể VectorRecord (id, values, norm)
│   │   └── action.py              # Thực thể ActionRecord (phục vụ Undo)
│   ├── core/                      # Lõi xử lý logic và thuật toán
│   │   ├── algorithms/            # 4 thuật toán tìm kiếm vector
│   │   │   ├── base.py            # Giao diện cơ sở BaseVectorIndex
│   │   │   ├── linear_scan.py     # 01. Linear Scan Baseline (Ground Truth)
│   │   │   ├── kdtree.py          # 02. KD-Tree (Spatial Indexing)
│   │   │   ├── ivf.py             # 03. IVF-Flat (Inverted File Index)
│   │   │   └── lsh.py             # 04. Random Hyperplane LSH
│   │   ├── structures/            # Cấu trúc dữ liệu tự cài đặt thuần túy
│   │   │   ├── heap.py            # BoundedMaxHeap (Top-K selection)
│   │   │   ├── stack.py           # LinkedStack (Undo)
│   │   │   └── queue.py           # LinkedQueue (Task Queue)
│   │   ├── storage.py             # I/O đọc và ghi file dữ liệu
│   │   ├── metrics.py             # Độ đo khoảng cách (Cosine, Euclidean, Dot Product)
│   │   ├── benchmark.py           # Đo lường Latency, Recall, QPS
│   │   └── manager.py             # VectorEngineManager (Facade điều phối trung tâm)
│   ├── ui/                        # Giao diện người dùng Streamlit
│   └── app.py                     # Entrypoint khởi chạy ứng dụng
├── tests/                         # Bộ kiểm thử tự động toàn diện (276 tests)
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 2. Hướng dẫn cài đặt & thực thi

### Yêu cầu môi trường

- Python 3.12+
- Quản lý gói bằng `uv` (hoặc `pip`)

### Cài đặt các phụ thuộc

```bash
uv sync
```

### Khởi chạy ứng dụng Web (Streamlit UI)

```bash
uv run streamlit run src/app.py
```

### Chạy toàn bộ 276 bài kiểm thử tự động (Unit Tests)

```bash
uv run pytest
```

### Xác thực nhanh 4 thuật toán bằng CLI

```bash
uv run python -c "from src.core.algorithms import LinearScanIndex, KDTreeIndex, IVFIndex, LSHIndex; print('Successfully loaded 4 core algorithms!')"
```
