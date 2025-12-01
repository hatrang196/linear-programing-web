from linear_programing_solver import (
    chuyen_ve_dang_chuan,
    hinh_hoc,
    don_hinh,
    hai_pha,
    bland
)

def solve_lp(A, b, c, loai, rls, var_types, method="Simplex"):
    try:
        # Bước 1: Chuẩn hóa dữ liệu (Bắt buộc phải có bước này)
        # Hàm gốc trả về: c_std, A_std, b_std, new_n_vars, standardized_var_names
        c_std, A_std, b_std, n_std, var_names = chuyen_ve_dang_chuan(loai, c, A, b, rls, var_types)
        
        # Bước 2: Gọi thuật toán tương ứng
        if method == "Geometric":
            # Hàm hình học cần var_names để vẽ
            final_x, z = hinh_hoc(A, b, c, loai, rls, var_types, var_names)
            return f"Kết quả (Hình học):\nNghiệm: {final_x}\nGiá trị Z: {z}"

        elif method == "Simplex":
            # Hàm đơn hình cần dữ liệu đã chuẩn hóa (std)
            final_x, z = don_hinh(c_std, A_std, b_std, loai, len(c), var_types, var_names)
            if final_x is None: return "Bài toán vô nghiệm hoặc không giới nội."
            return f"Kết quả (Đơn hình):\nNghiệm: {final_x}\nGiá trị Z: {z}"

        elif method == "Bland":
            final_x, z = bland(c_std, A_std, b_std, loai, len(c), var_types, var_names)
            if final_x is None: return "Bài toán vô nghiệm hoặc không giới nội."
            return f"Kết quả (Bland):\nNghiệm: {final_x}\nGiá trị Z: {z}"

        elif method == "Two-phase":
            # Phương pháp 2 pha hiện tại chỉ in log, chưa return string
            hai_pha(A_std, b_std, c_std, loai)
            return "Đã chạy xong 2 pha. Vui lòng kiểm tra 'Manage App -> Logs' để xem chi tiết các bảng đơn hình."

        return "Thuật toán chưa được hỗ trợ."

    except Exception as e:
        return f"Lỗi tính toán: {str(e)}"
