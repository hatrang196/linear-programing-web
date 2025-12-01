import streamlit as st
from lp_wrapper import solve_lp
import numpy as np

st.set_page_config(page_title="Giải Quy hoạch tuyến tính", layout="centered")

st.title("📘 Giải bài toán Quy hoạch tuyến tính")

st.write("""
Nhập dữ liệu hệ LP dưới dạng ma trận.
- A: Ma trận hệ số ràng buộc  
- b: vector ràng buộc  
- c: vector hệ số hàm mục tiêu  
""")

# SỬA 1: Chọn thuật toán
method = st.selectbox(
    "Chọn thuật toán:",
    ["Simplex", "Two-phase", "Bland", "Geometric"]
)

st.subheader("Nhập dữ liệu:")
# Input A
A_text = st.text_area("Ma trận A (các số cách nhau bởi dấu phẩy, xuống dòng là hàng mới):", "1, 2\n3, 1")

# Input b
b_text = st.text_area("Vector b (xuống dòng cho mỗi giá trị):", "10\n15")

# Input c
c_text = st.text_area("Vector c (ngăn cách bởi dấu phẩy):", "3, 4")

# SỬA 2: Dùng Selectbox thay vì Number Input để tránh lỗi int/string
loai_option = st.selectbox("Loại bài toán:", ["Max", "Min"])
loai = loai_option.lower() # Chuyển thành "max" hoặc "min"

# Input dấu
rls_text = st.text_area("Loại dấu ràng buộc (cách nhau bởi khoảng trắng, vd: <= <=):", "<= <=")

# Input loại biến
var_text = st.text_area("Loại biến (cách nhau bởi khoảng trắng, vd: >= >=):", ">= >=")

if st.button("🚀 Giải bài toán"):
    try:
        # Xử lý dữ liệu đầu vào
        A = np.array([list(map(float, row.split(","))) for row in A_text.strip().split("\n")])
        b = np.array([float(x) for x in b_text.strip().split("\n")])
        c = np.array([float(x) for x in c_text.strip().split(",")])

        rls = rls_text.strip().split()
        var_types = var_text.strip().split()

        # Gọi hàm giải
        result = solve_lp(A, b, c, loai, rls, var_types, method)

        # Hiển thị kết quả
        if "Lỗi" in result or "không" in result:
             st.warning(result)
        else:
             st.success("Kết quả tính toán:")
             st.code(result)

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu đầu vào: {e}")
