import streamlit as st
from core.manager import VectorEngineManager

st.set_page_config(
    page_title="Hệ thống Vector Search Engine",
    layout="wide"
)

# Duy trì trạng thái manager qua các lần tương tác
if "manager" not in st.session_state:
    st.session_state.manager = VectorEngineManager()

manager: VectorEngineManager = st.session_state.manager

st.title("Hệ thống Vector Search Engine (Mini FAISS / Milvus)")
st.caption("Đồ án Module 1 - DUT AI")

tab_data, tab_search = st.tabs([
    "Quản lý dữ liệu vector",
    "Truy vấn ngữ nghĩa và đo hiệu năng"
])

# TAB 1: QUẢN LÝ DỮ LIỆU
with tab_data:
    st.subheader("Danh sách vector hiện có")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Tổng số vector", manager.record_count)
    with col_stat2:
        st.metric("Số chiều (Dimension)", manager.dimension)
    with col_stat3:
        if st.button("Hoàn tác thao tác gần nhất (Undo)", use_container_width=True):
            msg = manager.undo()
            if msg:
                st.success(msg)
                st.rerun()
            else:
                st.info("Không có thao tác nào để hoàn tác.")

    all_records = manager.get_all_records()
    if all_records:
        table_rows = []
        for rec in all_records:
            vector_preview = ", ".join(f"{v:.2f}" for v in rec.values[:6])
            if rec.dimension > 6:
                vector_preview += "..."
            table_rows.append({
                "Mã ID": rec.id,
                "Từ vựng (Label)": rec.label,
                "Vector": f"[{vector_preview}]",
                "Chuẩn L2 (Norm)": f"{rec.norm:.4f}"
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.warning("Hệ thống chưa có bản ghi vector nào.")

    st.divider()
    col_add, col_del = st.columns(2)

    with col_add:
        st.markdown("### Thêm vector mới")
        with st.form("form_add_vector"):
            new_label = st.text_input("Từ vựng (Label):", placeholder="Ví dụ: computer")
            default_vals = " ".join(["0.50"] * manager.dimension) if manager.dimension > 0 else "0.1 0.2 0.3 0.4 0.5"
            new_values_raw = st.text_input(f"Tọa độ vector ({manager.dimension} chiều):", value=default_vals)
            submit_add = st.form_submit_button("Thêm bản ghi", use_container_width=True)

            if submit_add:
                try:
                    vals = [float(x.strip()) for x in new_values_raw.replace(",", " ").split() if x.strip()]
                    manager.add_record(label=new_label, values=vals)
                    st.success(f"Đã thêm vector '{new_label}'.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Lỗi: {str(ex)}")

    with col_del:
        st.markdown("### Xóa vector theo từ vựng")
        with st.form("form_delete_vector"):
            available_labels = [r.label for r in all_records]
            label_to_delete = st.selectbox("Chọn từ cần xóa:", options=available_labels if available_labels else [""])
            submit_del = st.form_submit_button("Xóa bản ghi", use_container_width=True)

            if submit_del and label_to_delete:
                deleted = manager.delete_record(label=label_to_delete)
                if deleted:
                    st.success(f"Đã xóa vector '{label_to_delete}'.")
                    st.rerun()
                else:
                    st.error("Không tìm thấy bản ghi.")

# TAB 2: TRUY VẤN VÀ ĐO HIỆU NĂNG
with tab_search:
    st.subheader("Truy vấn vector tương đồng (KNN Search)")
    col_query_type, col_metric_type = st.columns(2)
    with col_query_type:
        query_mode = st.radio("Phương thức truy vấn:", ["Chọn từ có sẵn", "Nhập vector tùy ý"], horizontal=True)

    with col_metric_type:
        metric_choice = st.radio("Độ đo:", ["Cosine Similarity", "Euclidean Distance"], horizontal=True)
        metric_key = "cosine" if "Cosine" in metric_choice else "euclidean"

    query_values = None
    query_name = ""

    if query_mode == "Chọn từ có sẵn":
        available_labels = [r.label for r in manager.get_all_records()]
        if available_labels:
            selected_word = st.selectbox("Chọn từ truy vấn:", options=available_labels)
            rec = manager.get_record(selected_word)
            if rec:
                query_values = rec.values
                query_name = rec.label
    else:
        raw_input = st.text_input(f"Nhập tọa độ ({manager.dimension} chiều):", value=" ".join(["0.8"] * manager.dimension))
        try:
            query_values = [float(x.strip()) for x in raw_input.replace(",", " ").split() if x.strip()]
            query_name = "Custom Vector"
        except Exception:
            st.error("Dữ liệu không hợp lệ.")

    max_k = max(1, manager.record_count)
    top_k = st.slider("Số lượng kết quả (Top-K):", min_value=1, max_value=max_k, value=min(5, max_k))

    if st.button("Thực thi tìm kiếm và đối chiếu thuật toán", type="primary", use_container_width=True):
        if query_values is None or len(query_values) != manager.dimension:
            st.error(f"Vector phải có đúng {manager.dimension} chiều.")
        else:
            linear_res, linear_ms = manager.search_linear(query_values, top_k=top_k, metric=metric_key)
            kdtree_res, kdtree_ms = manager.search_kdtree(query_values, top_k=top_k, metric=metric_key)

            st.markdown("### Kết quả đo thời gian thực thi (Benchmark)")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Linear Scan (O(N*d))", f"{linear_ms:.4f} ms")
            with b2:
                st.metric("KD-Tree Index (O(log N))", f"{kdtree_ms:.4f} ms")
            with b3:
                ratio = (linear_ms / kdtree_ms) if kdtree_ms > 0 else 1.0
                st.metric("Tỉ lệ tối ưu", f"{ratio:.2f}x")

            st.markdown(f"### Danh sách Top-{top_k} tương đồng với '{query_name}'")
            score_header = "Độ tương đồng Cosine" if metric_key == "cosine" else "Khoảng cách Euclidean"
            result_rows = []
            for rank, (record, score) in enumerate(linear_res, start=1):
                result_rows.append({
                    "Thứ hạng": rank,
                    "Từ vựng": record.label,
                    "Mã ID": record.id,
                    score_header: f"{score:.6f}",
                    "Vector": f"[{', '.join(f'{v:.2f}' for v in record.values[:5])}...]"
                })
            st.dataframe(result_rows, use_container_width=True, hide_index=True)