# ĐỒ ÁN MODULE 1: HỆ THỐNG VECTOR SEARCH ENGINE (MINI FAISS / MILVUS)

## 1. Cấu trúc thư mục dự án

```text
├── data/
│   ├── .gitkeep
│   └── vectors.txt                # Dữ liệu từ vựng và vector 50 chiều (chuẩn Stanford GloVe)
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Lớp trừu tượng BaseEntity
│   │   ├── record.py              # Lớp VectorRecord (thực thể vector thuần túy)
│   │   └── action.py              # Lớp ActionRecord (lưu vết thao tác cho Undo)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── algorithms/            # 5 thuật toán tìm kiếm vector
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Giao diện cơ sở BaseVectorIndex
│   │   │   ├── linear_scan.py     # Thuật toán 1: Exact Linear Scan (Baseline / Ground Truth)
│   │   │   ├── kdtree.py          # Thuật toán 2: KD-Tree (Phân vùng không gian theo trục)
│   │   │   ├── lsh.py             # Thuật toán 3: Random Hyperplane LSH (Băm nhạy cảm vị trí)
│   │   │   ├── ivf.py             # Thuật toán 4: IVF-Flat (Chỉ mục đảo kết hợp K-Means)
│   │   │   └── hnsw.py            # Thuật toán 5: HNSW (Đồ thị thế giới nhỏ phân tầng)
│   │   ├── structures/            # Các cấu trúc dữ liệu tự cài đặt thủ công
│   │   │   ├── __init__.py
│   │   │   ├── stack.py           # LinkedStack phục vụ hoàn tác (Undo)
│   │   │   ├── queue.py           # LinkedQueue phục vụ hàng đợi tác vụ truy vấn
│   │   │   └── heap.py            # BoundedMaxHeap phục vụ lấy Top-K độ phức tạp O(log K)
│   │   ├── storage.py             # Đọc và ghi file dữ liệu data/vectors.txt
│   │   ├── metrics.py             # Các hàm đo khoảng cách (Cosine, Euclidean, Dot Product)
│   │   ├── benchmark.py           # Module đo lường đa chỉ số (Latency, Recall@K, QPS, Build Time)
│   │   └── manager.py             # VectorEngineManager điều phối trung tâm (Facade Pattern)
│   └── app.py                     # Giao diện Streamlit 2 Tab (Quản lý dữ liệu & Benchmark)
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 2. Hướng dẫn cài đặt và thực thi

### Yêu cầu môi trường
- Python 3.12+
- Trình quản lý gói `uv`

### Cài đặt môi trường
```bash
uv sync
```

### Khởi chạy ứng dụng giao diện
```bash
uv run streamlit run src/app.py
```

### Chạy kiểm thử xác thực 5 thuật toán
```bash
uv run python -c "from src.core.algorithms import LinearScanIndex, KDTreeIndex, LSHIndex, IVFIndex, HNSWIndex; print('Đã nạp thành công 5 thuật toán')"
```
