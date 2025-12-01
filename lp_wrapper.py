from linear_programing_solver import (
    chuyen_ve_dang_chuan,
    hinh_hoc,
    don_hinh,
    hai_pha,
    bland
)
import numpy as np

def solve_lp(A, b, c, loai, rls, var_types, method="Simplex"):
    """
    Hàm wrapper để kết nối Web UI với file xử lý thuật toán.
    """
    
    # Bước 1: Chuẩn bị dữ liệu (Quan trọng: Phải có bước này mới chạy được thuật toán)
    # Hàm chuyen_ve_dang_chuan trả về: c_std, A_std, b_std, new_n_vars, standardized_var_names
    try:
        c_std, A_std, b_std, n_std, var_names = chuyen_ve_dang_chuan(loai, c, A, b, rls, var_types)
    except Exception as e:
        return f"Lỗi khi chuyển về dạng chuẩn: {str(e)}"

    # Bước 2: Gọi thuật toán tương ứng
    try:
        if method == "Geometric":
            # Hinh hoc can var_names de ve do thi
            final_x, z = hinh_hoc(A, b, c, loai, rls, var_types, var_names)
            return f"Kết quả (Hình học):\nNghiệm: {final_x}\nGiá trị Z: {z}"

        elif method == "Simplex":
            # Don hinh can du lieu da chuan hoa
            # Signature: don_hinh(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names)
            final_x, z = don_hinh(c_std, A_std, b_std, loai, len(c), var_types, var_names)
            
            if final_x is None and z is None:
                return "Bài toán không giới nội hoặc vô nghiệm (Kiểm tra log console để biết chi tiết)."
            return f"Kết quả (Đơn hình):\nNghiệm: {final_x}\nGiá trị Z: {z}"

        elif method == "Bland":
            # Bland tuong tu don hinh
            final_x, z = bland(c_std, A_std, b_std, loai, len(c), var_types, var_names)
            
            if final_x is None and z is None:
                return "Bài toán không giới nội hoặc vô nghiệm."
            return f"Kết quả (Bland):\nNghiệm: {final_x}\nGiá trị Z: {z}"

        elif method == "Two-phase":
            # Hai pha trong code cua ban hien tai chi in ra man hinh console (print)
            # chu chua return ket qua ve bien de hien thi len web.
            # Signature trong file: hai_pha(A, b, c, loai) -> Luu y thu tu tham so A, b, c
            # Code ban dang dung A_std cho hai pha
            
            # Lưu ý: Hàm hai_pha của bạn hiện tại không trả về (return) kết quả string mà chỉ print.
            # Để web hiển thị được, bạn cần sửa lại hàm hai_pha để return text, 
            # hoặc tạm thời chỉ báo thành công.
            hai_pha(A_std, b_std, c_std, loai) 
            return "Đã chạy phương pháp 2 pha. (Vui lòng xem kết quả chi tiết trong Console/Logs của ứng dụng vì hàm này chưa hỗ trợ trả về giao diện web)."

        return "Không nhận dạng được thuật toán."

    except Exception as e:
        return f"Đã xảy ra lỗi trong quá trình tính toán: {str(e)}"
