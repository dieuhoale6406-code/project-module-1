import streamlit as st

st.set_page_config(
    page_title="Đồ án Module 1",
    layout="wide"
)

st.title("Đồ án Module 1 - Xây dựng ứng dụng quản lý tương tác")

tab_data, tab_interactive = st.tabs([
    "Quản lý dữ liệu", 
    "Không gian tương tác"
])

with tab_data:
    st.subheader("Quản lý dữ liệu")
    st.write("Khu vực hiển thị danh sách và biểu mẫu nhập liệu.")

with tab_interactive:
    st.subheader("Không gian tương tác")
    st.write("Khu vực xử lý nghiệp vụ và cấu trúc dữ liệu.")
