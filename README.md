# ĐỒ ÁN MODULE 1: XÂY DỰNG ỨNG DỤNG QUẢN LÝ TƯƠNG TÁC BẰNG PYTHON

## 1. Cấu trúc thư mục

```text
4_ProjectModule/
├── data/
│   └── .gitkeep
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_structures.py
│   │   ├── storage.py
│   │   └── manager.py
│   └── app.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 2. Hướng dẫn cài đặt và chạy ứng dụng

Cài đặt môi trường và các gói phụ thuộc:
```bash
uv sync
```

Khởi chạy ứng dụng:
```bash
uv run streamlit run src/app.py
```
