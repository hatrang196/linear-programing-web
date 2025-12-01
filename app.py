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

method = st.selectbox(
    "Chọn thuật toán:",
    ["Geometric", "Simplex", "Two-phase", "Bland"]
)

st.subheader("Ma trận A (ngăn cách bằng dấu phẩy, mỗi dòng xuống hàng):")
A_text = st.text_area("A:", "1, 2\n3, 1")

b_text = st.text_area("Vector b:", "10\n15")
c_text = st.text_area("Vector c:", "3, 4")

loai = st.number_input("Loại bài toán (0 = max, 1 = min):", 0, 1, 0)

rls_text = st.text_area("Loại dấu ràng buộc (vd: <= <= >=):", "<= <=")
var_text = st.text_area("Loại biến (vd: >= >=):", ">= >=")

if st.button("🚀 Giải bài toán"):
    try:
        A = np.array([list(map(float, row.split(","))) for row in A_text.split("\n")])
        b = np.array([float(x) for x in b_text.split("\n")])
        c = np.array([float(x) for x in c_text.split(",")])

        rls = rls_text.split()
        var_types = var_text.split()

        result = solve_lp(A, b, c, loai, rls, var_types, method)

        st.success(result)

    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")
