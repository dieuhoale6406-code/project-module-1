# ĐỒ ÁN MODULE 1: HỆ THỐNG VECTOR SEARCH ENGINE (MINI FAISS / MILVUS)

> **Xây dựng Hệ thống Quản trị & Tìm kiếm Vector Tương đồng Bằng Cấu trúc Dữ liệu & Giải thuật Thuần túy**  
> Mô phỏng nguyên lý hoạt động của hai hệ thống tìm kiếm vector hàng đầu thế giới: **Meta FAISS** và **Milvus**.

---

## 1. Giới thiệu tổng quan & Ràng buộc kỹ thuật

Hệ thống cho phép nạp, quản lý vòng đời vector nhúng từ vựng (Stanford GloVe 50D), hỗ trợ giao dịch hoàn tác (Undo/Redo), thực thi 4 thuật toán tìm kiếm vector từ chính xác đến xấp xỉ (KNN / ANN), và đo lường hiệu năng đa chỉ số tự động.

### Ràng buộc công nghệ nghiêm ngặt:
* **Môi trường thực thi**: Python 3.12+, quản lý phụ thuộc bằng công cụ hiện đại `uv`.
* **Giao diện người dùng**: Streamlit Web Dashboard (hỗ trợ chuyển đổi giao diện Sáng / Tối).
* **Quy tắc thư viện bất biến**:
  * **CẤM** sử dụng các thư viện tính toán/giải thuật chuyên dụng bên ngoài (`numpy`, `scipy`, `scikit-learn`, `faiss`).
  * **CẤM** sử dụng các cấu trúc dữ liệu dựng sẵn của Python cho thuật toán (`heapq`, `collections.deque`).
  * Toàn bộ 4 thuật toán và 3 cấu trúc dữ liệu cốt lõi đều được tự cài đặt từ đầu bằng Python thuần túy.

---

## 2. Kết quả thực nghiệm Benchmark (Quy mô chuẩn N = 5.000)

Dưới đây là bảng tổng hợp đối sánh hiệu năng thực nghiệm giữa 4 thuật toán trên tập $N = 5.000$ vector ($d = 50$, Top-$K=5$, độ đo khoảng cách Euclidean):

| Thuật toán | Cấu hình tham số | Thời gian dựng (Build) | Thời gian tìm kiếm (Latency) | Tốc độ tăng tốc (Speedup) | Thông lượng (Throughput) | Độ chính xác (Recall@5) | Bản chất thuật toán |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Scan** | Baseline | $1.61\text{ ms}$ | $20.54\text{ ms}$ | **$1.0\text{x}$ (Chuẩn)** | $49\text{ QPS}$ | **$100.0\%$ (GT)** | Duyệt cạn toàn bộ $100\%$ dữ liệu. |
| **KD-Tree** | Cây trực giao $d=50$ | $61.52\text{ ms}$ | $14.70\text{ ms}$ | **$1.4\text{x}$** | $68\text{ QPS}$ | **$100.0\%$** | Chính xác tuyệt đối, backtrack nhiều do $d=50$. |
| **IVF-Flat** | $c=8\text{ cụm}, p=2\text{ probes}$ | $2,173.20\text{ ms}$ | $5.93\text{ ms}$ | **$3.5\text{x}$** | $169\text{ QPS}$ | **$97.0\%$** | Gom cụm Voronoi, chỉ quét cụm gần nhất. |
| **LSH** | $m=8\text{ bits}, L=2\text{ tables}$ | $45.00\text{ ms}$ | $2.20\text{ ms}$ | **$9.3\text{x}$** | $455\text{ QPS}$ | **$76.0\%$** | Băm ngẫu nhiên SimHash, giảm ứng viên tối đa. |

---

## 3. Cấu trúc thư mục dự án

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
│   │   │   ├── lsh.py             # 03. Random Hyperplane LSH (Multi-Table)
│   │   │   └── ivf.py             # 04. IVF-Flat (Inverted File Index)
│   │   ├── structures/            # Cấu trúc dữ liệu tự cài đặt thuần túy
│   │   │   ├── heap.py            # BoundedMaxHeap (Top-K selection)
│   │   │   ├── stack.py           # LinkedStack (Undo)
│   │   │   └── queue.py           # LinkedQueue (Task Queue)
│   │   ├── storage.py             # I/O đọc và ghi file dữ liệu (Multiprocessing)
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

## 4. Hướng dẫn cài đặt & Thực thi

### Yêu cầu môi trường

- Python 3.12+
- Quản lý gói bằng `uv` (khuyến nghị) hoặc `pip`

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
