import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from shapely.geometry import Polygon, Point, LineString
from scipy.optimize import linprog

# Ham ban dau
def nhap_bai_toan():
    """
    Hàm này cho phép người dùng nhập các thông số của bài toán quy hoạch tuyến tính.
    Bao gồm số biến, số ràng buộc, loại bài toán (tối đa/tối thiểu), hệ số hàm mục tiêu và các ràng buộc.
    """
    print("=== Nhập bài toán quy hoạch tuyến tính ===")

    while True:
        try:
            n = int(input("Nhập số biến (n, phải >= 2): "))
            if n < 2:
                print("Số biến phải từ 2 trở lên để có thể giải. Vui lòng nhập lại.")
            else:
                break
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập một số nguyên.")

    while True:
        try:
            m = int(input("Nhập số ràng buộc (m): "))
            if m <= 0:
                print("Số ràng buộc phải là một số nguyên dương. Vui lòng nhập lại.")
            else:
                break
        except ValueError:
            print("Lỗi: Số ràng buộc phải là một số nguyên. Vui lòng nhập lại.")

    loai_bt = input("Bài toán là TỐI ĐA (max) hay TỐ THIỂU (min)? [max/min]: ").strip().lower()
    while loai_bt not in ['max', 'min']:
        loai_bt = input("Vui lòng nhập lại (max/min): ").strip().lower()

    print("\nNhập hệ số hàm mục tiêu Z = c1*x1 + c2*x2 + ...:")
    while True:
        c_str = input(f"Hệ số c (cách nhau bởi dấu cách, có {n} hệ số): ").split()
        if len(c_str) != n:
            print(f"Phải nhập đúng {n} hệ số. Vui lòng nhập lại.")
        else:
            try:
                c = list(map(float, c_str))
                break
            except ValueError:
                print("Hệ số phải là số. Vui lòng nhập lại.")

    A = []
    b = []
    rls = []  # Dấu ràng buộc: '<=', '>=', '='

    print("\nNhập từng ràng buộc dạng: a1 a2 ... [dấu] b")
    print("Ví dụ: x1 + 0x2 +2x3 <= 10")
    print("Nhập là: 1 0 2 <= 10 (phải nhập đúng {n} hệ số a)\n")

    for i in range(m):
        while True:
            parts = input(f"Ràng buộc {i+1}: ").split()
            if len(parts) != n + 2:
                print(f"Phải nhập đúng {n} hệ số, [dấu] và 1 số b. Chú ý có khoảng cách. Vui lòng nhập lại.")
                continue
            *a_coeffs_str, op, b_val_str = parts
            if op not in ['<=', '>=', '=']:
                print("Dấu ràng buộc phải là '<=', '>=', hoặc '='. Nhập lại.")
                continue
            try:
                a_coeffs = list(map(float, a_coeffs_str))
                b_val = float(b_val_str)
                break
            except ValueError:
                print("Hệ số và b phải là số. Nhập lại.")

        A.append(a_coeffs)
        rls.append(op)
        b.append(b_val)

    # Nhập ràng buộc dấu cho từng xj
    var_types = []
    print("\nNhập ràng buộc dấu cho từng xj:")
    print(" - Nhập là: '>=' cho xj >= 0")
    print(" - Nhập là: '<=' cho xj <= 0")
    print(" - Nhập là: 'free' cho xj tự do")
    for i in range(n):
        while True:
            var_type = input(f"Ràng buộc dấu của x{i+1} [>=/<=/free]: ").strip().lower()
            if var_type in ['>=', '<=', 'free']:
                var_types.append(var_type)
                break
            else:
                print("Không hợp lệ. Vui lòng nhập: '<=', '>=', hoặc 'free'. Nhập lại.")

    return loai_bt, np.array(c), np.array(A, dtype=float), np.array(b, dtype=float), rls, n, var_types

def chuyen_ve_dang_chuan(loai_bt, c, A, b, rls, var_types):
    """
    Chuyển bài toán quy hoạch tuyến tính về dạng chuẩn (min C^T x , Ax <= b, x >= 0).
    Nếu là bài toán max, đổi dấu hàm mục tiêu.
    Nếu có ràng buộc '>=', đổi dấu cả hàng.
    Nếu có ràng buộc '=', tách thành hai ràng buộc '<=' và '>='.
    Xử lý biến tự do và biến không dương.
    """
    print("******************************************************")

    original_n_vars = c.shape[0]

    # Bước 1: Xử lý hàm mục tiêu (chuyển về min)
    if loai_bt == 'max':
        c_std = -c  # Chuyển bài toán max Z thành min -Z
    else:
        c_std = np.copy(c) # Tạo bản sao để không thay đổi c gốc

    # Bước 2: Xử lý biến tự do và biến không dương
    new_n_vars = 0
    variable_transformations = []
    standardized_var_names = [] # Danh sách các tên biến mới sau khi chuẩn hóa

    # Tính toán số biến mới sau khi chuẩn hóa
    for i in range(original_n_vars):
        if var_types[i] == 'free':
            new_n_vars += 2
        else: # '>= 0' hoặc '<= 0'
            new_n_vars += 1

    # Khởi tạo ma trận A_new và vector c_new với kích thước mới
    c_new = np.zeros(new_n_vars)
    A_new = np.zeros((A.shape[0], new_n_vars))

    current_original_col = 0 # Chỉ số cột trong ma trận A gốc
    current_new_col = 0      # Chỉ số cột trong ma trận A_new

    for i in range(original_n_vars):
        if var_types[i] == 'free':
            # x_i = x_i^+ - x_i^- với x_i^+, x_i^- >= 0
            variable_transformations.append(f"x{i+1} = x{i+1}^+ - x{i+1}^-")
            standardized_var_names.append(f"x{i+1}^+")
            standardized_var_names.append(f"x{i+1}^-")

            c_new[current_new_col] = c_std[current_original_col]
            c_new[current_new_col + 1] = -c_std[current_original_col]
            A_new[:, current_new_col] = A[:, current_original_col]
            A_new[:, current_new_col + 1] = -A[:, current_original_col]

            current_new_col += 2
        elif var_types[i] == '<=': # Lưu ý: nhập từ người dùng là '<=', không phải '<= 0'
            # y_i = -x_i với y_i >= 0 => x_i = -y_i
            variable_transformations.append(f"y{i+1} = -x{i+1}")
            standardized_var_names.append(f"y{i+1}")

            c_new[current_new_col] = -c_std[current_original_col]
            A_new[:, current_new_col] = -A[:, current_original_col]

            current_new_col += 1
        else: # '>=' (giữ nguyên x_i >= 0)
            variable_transformations.append(f"x{i+1} (giữ nguyên)")
            standardized_var_names.append(f"x{i+1}")

            c_new[current_new_col] = c_std[current_original_col]
            A_new[:, current_new_col] = A[:, current_original_col]

            current_new_col += 1
        current_original_col += 1 # Tăng chỉ số biến gốc sau mỗi lần xử lý

    # In ra các phép thế biến
    if variable_transformations:
        print("\nPhép thế biến:")
        for trans in variable_transformations:
            print(f" Đặt {trans}")

    c_std = c_new
    A_std = A_new

    # Bước 3: Xử lý các ràng buộc
    A_final_list = []
    b_final_list = []

    for i in range(len(rls)):
        if rls[i] == '<=':
            A_final_list.append(A_std[i])
            b_final_list.append(b[i])
        elif rls[i] == '>=':  # Đổi dấu để thành <=
            A_final_list.append(-A_std[i])
            b_final_list.append(-b[i])
        elif rls[i] == '=':   # Tách thành 2 ràng buộc <= và >=
            A_final_list.append(A_std[i])
            b_final_list.append(b[i])
            A_final_list.append(-A_std[i])
            b_final_list.append(-b[i])
        else:
            raise ValueError(f"Dấu ràng buộc không hợp lệ: {rls[i]}")

    A_std_final = np.array(A_final_list, dtype=float)
    b_std_final = np.array(b_final_list, dtype=float)

    # In bài toán đã chuyển về dạng chuẩn
    print("---- Bài toán sau khi đã chuyển về dạng chuẩn ----")

    z_str = "min Z = "
    for i in range(len(c_std)):
        if abs(c_std[i]) > 1e-10: # Kiểm tra giá trị khác 0 đáng kể
            sign = "+" if c_std[i] > 0 else "-"
            if i == 0: # Số hạng đầu tiên không cần dấu '+' nếu nó dương
                sign = "-" if c_std[i] < 0 else ""
            z_str += f"{sign} {abs(c_std[i]):.4f}{standardized_var_names[i]}"

    print(z_str.replace("+-", "- ").replace("++", "+ ").strip()) # Clean up double signs

    print("\nRàng buộc:")
    for i in range(A_std_final.shape[0]):
        constraint_str = ""
        for j in range(A_std_final.shape[1]):
            coeff = A_std_final[i, j]
            if abs(coeff) > 1e-10: # Kiểm tra giá trị khác 0 đáng kể
                sign = "+" if coeff > 0 else "-"
                if j == 0: # Số hạng đầu tiên không cần dấu '+' nếu nó dương
                    sign = "-" if coeff < 0 else ""
                constraint_str += f"{sign} {abs(coeff):.4f}{standardized_var_names[j]}"

        constraint_str = constraint_str.strip().replace("+-", "- ")
        if constraint_str.startswith("+"):
            constraint_str = constraint_str[1:].strip()
        print(f"{i+1}. {constraint_str} <= {b_std_final[i]:.4f}")

    print("\nRàng buộc dấu:")
    for i in range(new_n_vars):
        print(f"{standardized_var_names[i]} >= 0")

    return c_std, A_std_final, b_std_final, new_n_vars, standardized_var_names

def xet_phuong_phap(n, b_std):
    """
    Xác định các phương pháp giải phù hợp dựa trên số biến và giá trị b_std,
    sau đó cho phép người dùng lựa chọn. Phương pháp hình học chỉ áp dụng cho 2 biến.
    """
    # Định nghĩa các phương pháp:
    # 1: PP hình học (chỉ giải cho trường hợp 2 biến)
    # 2: PP đơn hình
    # 3: PP Bland
    # 4: PP hai pha

    print("------ Lựa chọn phương pháp ------")

    if n == 2:
        if np.any(b_std < -1e-9): # Nếu có b_i âm sau chuẩn hóa
            print("\nCác phương pháp gợi ý cho bài toán 2 biến và có b < 0:")
            print(" 1 - Phương pháp hình học")
            print(" 4 - Phương pháp hai pha")
            valid_choices = [1, 4]
        elif np.any(np.abs(b_std) < 1e-9): # Nếu có b_i bằng 0 sau chuẩn hóa
            print("\nCác phương pháp gợi ý cho bài toán 2 biến và có b = 0:")
            print(" 1 - Phương pháp hình học")
            print(" 3 - Phương pháp Bland")
            valid_choices = [1, 3]
        else: # Các trường hợp còn lại (b_std > 0)
            print("\nCác phương pháp gợi ý cho bài toán 2 biến và b > 0:")
            print(" 1 - Phương pháp hình học")
            print(" 2 - Phương pháp đơn hình")
            print(" 3 - Phương pháp Bland")
            valid_choices = [1, 2, 3]
    else: # n > 2
        if np.any(b_std < -1e-9): # Nếu có b_i âm sau chuẩn hóa
            print("\nVới nhiều hơn 2 biến và có b < 0, phương pháp phù hợp nhất là:")
            print(" 4 - Phương pháp hai pha")
            valid_choices = [4]
        elif np.any(np.abs(b_std) < 1e-9): # Nếu có b_i bằng 0 sau chuẩn hóa
            print("\nVới nhiều hơn 2 biến và có b = 0, các phương pháp gợi ý:")
            print(" 3 - Phương pháp Bland")
            valid_choices = [3]
        else: # Các trường hợp còn lại (n > 2, b_std > 0)
            print("\nVới nhiều hơn 2 biến và b > 0, các phương pháp gợi ý:")
            print(" 2 - Phương pháp đơn hình")
            print(" 3 - Phương pháp Bland")
            valid_choices = [2, 3]

    # Vòng lặp để người dùng nhập lựa chọn hợp lệ
    while True:
        try:
            choice = int(input(f"Vui lòng nhập lựa chọn của bạn {valid_choices}: "))
            if choice in valid_choices:
                return choice
            else:
                print(f"Lựa chọn không hợp lệ. Vui lòng nhập một trong các số sau: {valid_choices}")
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập một số nguyên.")

def khoi_tao_bang_tu_vung(c_std, A_std, b_std):
    """
    Khởi tạo bảng từ vựng ban đầu.
    Thêm các biến bù vào ma trận A và tạo hàng mục tiêu (hàng 0).
    """
    m_std, n_original_vars = A_std.shape

    I = np.eye(m_std)

    B = np.zeros((m_std + 1, 1 + n_original_vars + m_std))

    B[0, 0] = 0.0
    B[0, 1:1 + n_original_vars] = c_std

    B[1:, 0] = b_std
    B[1:, 1:1 + n_original_vars] = A_std
    B[1:, 1 + n_original_vars:] = I

    co_so = [n_original_vars + i for i in range(m_std)]

    return B, co_so

def in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_standardized_vars, is_optimal_tableau=False):
    """
    In bảng từ vựng ra màn hình.
    Hiển thị rõ ràng các biến cơ sở, biến không cơ sở, hệ số và tỉ lệ.
    """
    m_std = B.shape[0] - 1

    if is_optimal_tableau:
        print("\n--- Từ vựng tối ưu ---")
    else:
        print(f"\n--- Bảng từ vựng {pivot_count} ---")

    column_headers = bien_names[:n_standardized_vars] + \
                     [f"w{i+1}" for i in range(m_std)] + [' ']

    header_line = " " * 13
    for h in column_headers:
        header_line += f"{h:>10}"
    print(header_line)
    print("-" * (13 + 10 * len(column_headers)))

    line = f"{'z':<12}|"
    for j in range(1, B.shape[1]):
        line += f"{B[0, j]:10.3f}"
    line += f"{B[0, 0]:10.3f}"
    print(line)
    print("-" * (13 + 10 * len(column_headers)))

    ti_le_strs = []
    if bien_vao_col_idx != -1:
        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10:
                ratio = B[i, 0] / a_ij
                ti_le_strs.append(f"({B[i,0]:.3f}/{a_ij:.3f}={ratio:.3f})")
            else:
                ti_le_strs.append("   ")
    else:
        ti_le_strs = [""] * m_std

    for i in range(1, m_std + 1):
        var_idx_in_basis = co_so[i-1]
        row_name = bien_names[var_idx_in_basis]

        line = f"{row_name:<12}|"
        for j in range(1, B.shape[1]):
            line += f"{B[i, j]:10.3f}"
        line += f"{B[i, 0]:10.3f}     | {ti_le_strs[i-1]}"
        print(line)
    print("-" * (13 + 10 * len(column_headers)))



# Phuong phap hinh hoc
def hinh_hoc(A, b, c, loai, rls, var_types, standardized_var_names):
    """
    Giải bài toán quy hoạch tuyến tính 2 biến bằng phương pháp hình học.

    Args:
        A (np.array): Ma trận hệ số của các ràng buộc.
        b (np.array): Vector vế phải của các ràng buộc.
        c (np.array): Vector hệ số hàm mục tiêu.
        loai (str): 'max' hoặc 'min' (tối đa hoặc tối thiểu).
        rls (list): Danh sách các dấu ràng buộc ('<=', '>=', '=').
        var_types (list): Ràng buộc dấu của các biến ('free', '>=', '<=').
        standardized_var_names (list): Tên các biến đã chuẩn hóa.
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    n_vars = A.shape[1]

    if n_vars != 2:
        print("Phương pháp hình học chỉ áp dụng cho bài toán 2 biến.")
        return None, None

    # --- Điều chỉnh phạm vi vẽ đồ thị ban đầu dựa trên var_types ---
    x_min_plot_initial = -15
    x_max_plot_initial = 15
    y_min_plot_initial = -15
    y_max_plot_initial = 15

    if var_types[0] == '>=':
        x_min_plot_initial = 0
    elif var_types[0] == '<=':
        x_max_plot_initial = 0

    if var_types[1] == '>=':
        y_min_plot_initial = 0
    elif var_types[1] == '<=':
        y_max_plot_initial = 0

    plot_range_buffer = 50
    x_min_plot = min(x_min_plot_initial, -plot_range_buffer)
    x_max_plot = max(x_max_plot_initial, plot_range_buffer)
    y_min_plot = min(y_min_plot_initial, -plot_range_buffer)
    y_max_plot = max(y_max_plot_initial, plot_range_buffer)

    all_intersection_points = []
    lines = []

    for i in range(len(A)):
        line_a = A[i]
        line_b = b[i]

        if line_a[1] != 0:
            line = LineString([(x_val, (line_b - line_a[0]*x_val) / line_a[1]) for x_val in [-plot_range_buffer, plot_range_buffer]])
            lines.append(line)
        elif line_a[0] != 0:
            line = LineString([(line_b / line_a[0], y_val) for y_val in [-plot_range_buffer, plot_range_buffer]])
            lines.append(line)

    if var_types[0] != 'free':
        lines.append(LineString([(0, -plot_range_buffer), (0, plot_range_buffer)]))
    if var_types[1] != 'free':
        lines.append(LineString([(-plot_range_buffer, 0), (plot_range_buffer, 0)]))

    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i < j:
                intersection = line1.intersection(line2)
                if intersection.geom_type == 'Point':
                    all_intersection_points.append([intersection.x, intersection.y])
                elif intersection.geom_type == 'MultiPoint':
                    for pt in intersection.geoms:
                        all_intersection_points.append([pt.x, pt.y])

    # Lọc các điểm khả thi
    feasible_points = []
    for point_coords in all_intersection_points:
        point = np.array(point_coords)
        is_feasible = True
        for k in range(len(A)):
            val = np.dot(A[k], point)
            if rls[k] == '<=' and val > b[k] + 1e-7:
                is_feasible = False
                break
            elif rls[k] == '>=' and val < b[k] - 1e-7:
                is_feasible = False
                break
            elif rls[k] == '=' and abs(val - b[k]) > 1e-7:
                is_feasible = False
                break

        if var_types[0] == '>=' and point[0] < -1e-7:
            is_feasible = False
        elif var_types[0] == '<=' and point[0] > 1e-7:
            is_feasible = False

        if var_types[1] == '>=' and point[1] < -1e-7:
            is_feasible = False
        elif var_types[1] == '<=' and point[1] > 1e-7:
            is_feasible = False

        if is_feasible:
            feasible_points.append(point)

    # Loại bỏ các điểm trùng lặp gần nhau
    unique_feasible_points = []
    seen_points_tuples = set()
    for p in feasible_points:
        p_rounded = tuple(np.round(p, 6))
        if p_rounded not in seen_points_tuples:
            unique_feasible_points.append(p)
            seen_points_tuples.add(p_rounded)
    feasible_points = unique_feasible_points


    # Kiểm tra trạng thái bài toán bằng linprog trước
    A_ub_lp, b_ub_lp, A_eq_lp, b_eq_lp = [], [], [], []
    for i in range(len(A)):
        if rls[i] == '<=':
            A_ub_lp.append(A[i])
            b_ub_lp.append(b[i])
        elif rls[i] == '>=':
            A_ub_lp.append(-A[i])
            b_ub_lp.append(-b[i])
        elif rls[i] == '=':
            A_eq_lp.append(A[i])
            b_eq_lp.append(b[i])

    bounds_lp = []
    for i in range(n_vars):
        if var_types[i] == '>=':
            bounds_lp.append((0, None))
        elif var_types[i] == '<=':
            bounds_lp.append((None, 0))
        else: # 'free'
            bounds_lp.append((None, None))

    # Cần đảo dấu c nếu là bài toán 'max' vì linprog luôn tìm 'min'
    c_lp = c if loai == 'min' else -c

    res_check = linprog(
        c=c_lp,
        A_ub=np.array(A_ub_lp) if A_ub_lp else None,
        b_ub=np.array(b_ub_lp) if b_ub_lp else None,
        A_eq=np.array(A_eq_lp) if A_eq_lp else None,
        b_eq=np.array(b_eq_lp) if b_eq_lp else None,
        bounds=bounds_lp,
        method='highs'
    )

    # --- Xử lý các trường hợp đặc biệt: Vô nghiệm và Không giới nội ---
    if res_check.status == 2: # Infeasible (Vô nghiệm)
        print("\n⇒ Bài toán vô nghiệm (miền khả thi rỗng).")
        plt.figure(figsize=(8, 8))
        colors = cm.get_cmap('tab10').resampled(len(A))
        for i in range(len(A)):
            x_vals_plot = np.linspace(x_min_plot_initial, x_max_plot_initial, 400)
            if A[i][1] != 0:
                y_vals_plot = (b[i] - A[i][0] * x_vals_plot) / A[i][1]
                plt.plot(x_vals_plot, y_vals_plot, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}', linestyle='--', color=colors(i))
            elif A[i][0] != 0:
                plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}', linestyle='--', color=colors(i))
        if var_types[0] == '>=': plt.axvline(0, color='gray', linestyle='--', label='x1 >= 0')
        if var_types[0] == '<=': plt.axvline(0, color='gray', linestyle='--', label='x1 <= 0')
        if var_types[1] == '>=': plt.axhline(0, color='gray', linestyle='--', label='x2 >= 0')
        if var_types[1] == '<=': plt.axhline(0, color='gray', linestyle='--', label='x2 <= 0')

        plt.xlim(x_min_plot_initial, x_max_plot_initial)
        plt.ylim(y_min_plot_initial, y_max_plot_initial)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Bài toán vô nghiệm: Không có miền khả thi")
        plt.legend()
        plt.grid(True)
        plt.show()
        return None, None

    if res_check.status == 3: # Unbounded (Không giới nội)
        unbounded_value = "+∞" if loai == 'max' else "-∞"
        print(f"\n⇒ Bài toán không giới nội. Hàm mục tiêu đạt {loai} tại {unbounded_value}.")

        plt.figure(figsize=(8, 8))
        colors = matplotlib.colormaps['tab10'].resampled(len(A))
        for i in range(len(A)):
            x_vals_plot = np.linspace(x_min_plot_initial, x_max_plot_initial, 400)
            if A[i][1] != 0:
                y_vals_plot = (b[i] - A[i][0] * x_vals_plot) / A[i][1]
                plt.plot(x_vals_plot, y_vals_plot, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}', color=colors(i), linestyle='-')
            elif A[i][0] != 0:
                plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}', color=colors(i), linestyle='-')

        plt.xlim(x_min_plot_initial, x_max_plot_initial)
        plt.ylim(y_min_plot_initial, y_max_plot_initial)
        plt.axvline(0, color='black', linewidth=0.8, zorder=0)
        plt.axhline(0, color='black', linewidth=0.8, zorder=0)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Bài toán không giới nội")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        return None, None

    # Nếu có điểm khả thi, tiến hành tìm giá trị tối ưu và vẽ đồ thị
    # Phần này sẽ chỉ chạy nếu linprog không báo vô nghiệm hoặc không giới nội
    # và feasible_points không rỗng.
    if not feasible_points:
        # Trường hợp này có thể xảy ra nếu linprog không báo vô nghiệm,
        # nhưng quá trình tìm giao điểm và lọc lại không tìm được điểm nào.
        # Đây là một trường hợp "vô nghiệm" khác, có thể do lỗi làm tròn
        # hoặc miền khả thi rất nhỏ không có đỉnh rõ ràng trong phạm vi tìm kiếm.
        print("\n⇒ Bài toán vô nghiệm (Không có điểm khả thi sau khi lọc các giao điểm).")
        plt.figure(figsize=(8, 8))
        colors = cm.get_cmap('tab10').resampled(len(A))
        for i in range(len(A)):
            x_vals_plot = np.linspace(x_min_plot_initial, x_max_plot_initial, 400)
            if A[i][1] != 0:
                y_vals_plot = (b[i] - A[i][0] * x_vals_plot) / A[i][1]
                plt.plot(x_vals_plot, y_vals_plot, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}', linestyle='--', color=colors(i))
            elif A[i][0] != 0:
                plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}', linestyle='--', color=colors(i))
        if var_types[0] == '>=': plt.axvline(0, color='gray', linestyle='--', label='x1 >= 0')
        if var_types[0] == '<=': plt.axvline(0, color='gray', linestyle='--', label='x1 <= 0')
        if var_types[1] == '>=': plt.axhline(0, color='gray', linestyle='--', label='x2 >= 0')
        if var_types[1] == '<=': plt.axhline(0, color='gray', linestyle='--', label='x2 <= 0')

        plt.xlim(x_min_plot_initial, x_max_plot_initial)
        plt.ylim(y_min_plot_initial, y_max_plot_initial)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Bài toán vô nghiệm: Không có miền khả thi")
        plt.legend()
        plt.grid(True)
        plt.show()
        return None, None


    # Tính giá trị hàm mục tiêu tại các điểm khả thi
    Z_vals = [np.dot(c, x) for x in feasible_points]

    if loai == 'min':
        val_opt = min(Z_vals)
        cmp_func = lambda z: np.isclose(z, val_opt, atol=1e-6)
    else: # loai == 'max'
        val_opt = max(Z_vals)
        cmp_func = lambda z: np.isclose(z, val_opt, atol=1e-6)

    diem_toi_uu_all = []
    seen = set()
    for i in range(len(Z_vals)):
        if cmp_func(Z_vals[i]):
            point_rounded = tuple(np.round(feasible_points[i], 6))
            if point_rounded not in seen:
                seen.add(point_rounded)
                diem_toi_uu_all.append(feasible_points[i])

    final_x_values = None
    val_primal = val_opt

    if len(diem_toi_uu_all) == 1:
        final_x_values = diem_toi_uu_all[0]
    elif len(diem_toi_uu_all) > 1:
        diem_toi_uu_all_sorted = sorted(diem_toi_uu_all, key=lambda p: (p[0], p[1]))
        final_x_values = diem_toi_uu_all_sorted

    # Hiển thị kết quả
    print("\n--- Kết quả Phương pháp hình học ---")
    print("Các điểm khả thi:")
    for point, val in zip(feasible_points, Z_vals):
        print(f"x = ({point[0]:.4f}, {point[1]:.4f}), Z = {val:.4f}")

    if len(diem_toi_uu_all) > 1:
        print("\n⇒ Bài toán có vô số nghiệm tối ưu.")
        p1 = final_x_values[0]
        p2 = final_x_values[-1]
        print(f"Tập nghiệm tối ưu là đoạn thẳng từ ({p1[0]:.4f}, {p1[1]:.4f}) đến ({p2[0]:.4f}, {p2[1]:.4f})")
        print("⇒ Các nghiệm có dạng:")
        print(f"   x = (1 - t)*({p1[0]:.4f}, {p1[1]:.4f}) + t*({p2[0]:.4f}, {p2[1]:.4f}) với t ∈ [0, 1]")
    else:
        print("\n⇒ Bài toán có nghiệm tối ưu duy nhất.")

    print(f"Giá trị tối ưu Z = {val_primal:.4f}")

    if len(diem_toi_uu_all) == 1:
        print("Nghiệm tối ưu:")
        print(f"   x1 = {final_x_values[0]:.4f}, x2 = {final_x_values[1]:.4f}")


    # --- BẮT ĐẦU PHẦN VẼ ĐỒ THỊ MỚI VỚI FILLING HIỆU QUẢ HƠN ---
    plt.figure(figsize=(10, 8))

    # Điều chỉnh giới hạn trục để đảm bảo tất cả các điểm quan trọng hiển thị
    all_x = [p[0] for p in feasible_points] if feasible_points else [0]
    all_y = [p[1] for p in feasible_points] if feasible_points else [0]

    if final_x_values is not None:
        if isinstance(final_x_values, list): # Trường hợp vô số nghiệm
            all_x.extend([p[0] for p in final_x_values])
            all_y.extend([p[1] for p in final_x_values])
        elif isinstance(final_x_values, np.ndarray): # Trường hợp nghiệm duy nhất
            all_x.append(final_x_values[0])
            all_y.append(final_x_values[1])

    if var_types[0] == '>=': all_x.append(0)
    if var_types[0] == '<=': all_x.append(0)
    if var_types[1] == '>=': all_y.append(0)
    if var_types[1] == '<=': all_y.append(0)

    if all_x and all_y:
        x_min_data, x_max_data = min(all_x), max(all_x)
        y_min_data, y_max_data = min(all_y), max(all_y)

        x_buffer = max(1, (x_max_data - x_min_data) * 0.2)
        y_buffer = max(1, (y_max_data - y_min_data) * 0.2)

        final_x_min = min(x_min_plot_initial, x_min_data - x_buffer)
        final_x_max = max(x_max_plot_initial, x_max_data + x_buffer)
        final_y_min = min(y_min_plot_initial, y_min_data - y_buffer)
        final_y_max = max(y_max_plot_initial, y_max_data + y_buffer)
    else:
        final_x_min, final_x_max = x_min_plot_initial, x_max_plot_initial
        final_y_min, final_y_max = y_min_plot_initial, y_max_plot_initial

    num_points = 500
    x = np.linspace(final_x_min, final_x_max, num_points)
    y = np.linspace(final_y_min, final_y_max, num_points)
    X, Y = np.meshgrid(x, y)
    Z_feasible = np.ones(X.shape, dtype=bool)

    for i in range(len(A)):
        constraint_val = A[i][0] * X + A[i][1] * Y
        if rls[i] == '<=':
            Z_feasible = np.logical_and(Z_feasible, constraint_val <= b[i] + 1e-7)
        elif rls[i] == '>=':
            Z_feasible = np.logical_and(Z_feasible, constraint_val >= b[i] - 1e-7)
        elif rls[i] == '=':
            Z_feasible = np.logical_and(Z_feasible, np.isclose(constraint_val, b[i], atol=1e-2))

    if var_types[0] == '>=':
        Z_feasible = np.logical_and(Z_feasible, X >= -1e-7)
    elif var_types[0] == '<=':
        Z_feasible = np.logical_and(Z_feasible, X <= 1e-7)

    if var_types[1] == '>=':
        Z_feasible = np.logical_and(Z_feasible, Y >= -1e-7)
    elif var_types[1] == '<=':
        Z_feasible = np.logical_and(Z_feasible, Y <= 1e-7)

    plt.imshow(Z_feasible, origin='lower', extent=[final_x_min, final_x_max, final_y_min, final_y_max],
                cmap='Reds', alpha=0.2, aspect='auto', label='Miền khả thi')

    colors = cm.get_cmap('tab10').resampled(len(A))
    for i in range(len(A)):
        x_vals_line = np.linspace(final_x_min, final_x_max, 400)
        if A[i][1] != 0:
            y_vals_line = (b[i] - A[i][0] * x_vals_line) / A[i][1]
            plt.plot(x_vals_line, y_vals_line, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}',
                     color=colors(i), linestyle='-', linewidth=2, zorder=3)
        elif A[i][0] != 0:
            plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}',
                        color=colors(i), linestyle='-', linewidth=2, zorder=3)

    plt.axvline(0, color='black', linewidth=0.8, zorder=0)
    plt.axhline(0, color='black', linewidth=0.8, zorder=0)

    if feasible_points:
        xs, ys = zip(*feasible_points)
        plt.scatter(xs, ys, color='blue', zorder=4, label='Các đỉnh khả thi', s=50, edgecolors='black')

    if final_x_values is not None:
        if isinstance(final_x_values, list) and len(final_x_values) > 1:
            p1, p2 = final_x_values[0], final_x_values[-1]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=3, label='Đoạn nghiệm tối ưu', zorder=5)
            plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', marker='o', s=100, zorder=6, edgecolors='black')
        elif isinstance(final_x_values, np.ndarray):
            x_opt, y_opt = final_x_values[0], final_x_values[1]
            plt.scatter(x_opt, y_opt, color='red', marker='o', s=100, label='Nghiệm tối ưu', zorder=6, edgecolors='black')

    plt.xlim(final_x_min, final_x_max)
    plt.ylim(final_y_min, final_y_max)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Miền nghiệm và nghiệm tối ưu ({loai.upper()} Z)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return final_x_values, val_primal

# Phuong phap don hinh
def don_hinh(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names):
    """
    Giải bài toán quy hoạch tuyến tính bằng phương pháp đơn hình tiêu chuẩn.
    Hàm này thực hiện các bước pivot cho đến khi tìm được nghiệm tối ưu hoặc kết luận bài toán không giới nội.
    """
    m_std, n_std_vars = A_std.shape

    # Khởi tạo bảng từ vựng và danh sách biến cơ sở
    B, co_so = khoi_tao_bang_tu_vung(c_std, A_std, b_std)

    # bien_names là tên của tất cả các biến (biến gốc chuẩn hóa và biến bù)
    bien_names = standardized_var_names + [f"w{i+1}" for i in range(m_std)]

    pivot_count = 0 # Đếm số lần pivot

    print("\n--- Bắt đầu giải bằng phương pháp đơn hình tiêu chuẩn ---")

    # Kiểm tra điều kiện đầu vào: đơn hinh yêu cầu tất cả b_std phải không âm
    # để có thể bắt đầu từ một giải pháp cơ sở khả thi (các biến bù là biến cơ sở ban đầu).
    if np.any(b_std < -1e-9):
        print("\n!!! Lỗi: Phương pháp đơn hình tiêu chuẩn yêu cầu tất cả các giá trị vế phải (b_i) phải không âm.")
        print("!!! Bài toán của bạn có b_i âm sau khi chuyển về dạng chuẩn.")
        print("!!! Để giải bài toán này, bạn cần sử dụng Phương pháp Hai Pha hoặc biến đổi thêm để đảm bảo tính khả thi ban đầu.")
        print("Bài toán không thể giải được bằng Phương pháp đơn hình tiêu chuẩn theo cách này.")
        return None, None # Trả về None nếu không thể giải

    while True:
        pivot_count += 1
        bien_vao_col_idx = -1 # Chỉ số cột của biến vào (trong bảng B)
        max_negative_coeff = 0.0 # Để tìm hệ số âm lớn nhất (tức là âm nhất)

        # Bước 1: Chọn biến vào (theo đơn hinh: hệ số âm có giá trị tuyệt đối lớn nhất)
        # Duyệt qua các cột biến (từ x1...xn, w1...wm) trong hàng mục tiêu
        for j in range(1, 1 + n_std_vars + m_std):
            if B[0, j] < -1e-10: # Nếu hệ số trong hàng mục tiêu âm
                if B[0, j] < max_negative_coeff: # So sánh với hệ số âm lớn nhất đã tìm thấy
                    max_negative_coeff = B[0, j]
                    bien_vao_col_idx = j

        # Bước 2: Kiểm tra điều kiện dừng (tối ưu)
        if bien_vao_col_idx == -1:
            # Nếu không tìm thấy hệ số âm nào trong hàng mục tiêu, bài toán đã tối ưu
            in_bang_tu_vung(B, bien_names, co_so, -1, pivot_count, n_std_vars, is_optimal_tableau=True) # In bảng tối ưu

            # Giá trị tối ưu của hàm mục tiêu (Z) trong bài toán chuẩn (min -Z hoặc min Z)
            z_opt_std = -B[0, 0]
            # Chuyển về giá trị tối ưu của bài toán gốc (max Z hoặc min Z)
            val_primal = -z_opt_std if loai_bt == 'max' else z_opt_std

            # Bước 3: Kiểm tra vô số nghiệm tối ưu
            has_multiple_solutions = False
            non_basic_vars_with_zero_cost = []

            # Tạo danh sách biến tự do (biến có cặp ^+ và ^-)
            free_var_bases = set()
            for name in bien_names:
              if name.endswith('^+') or name.endswith('^-'):
                free_var_bases.add(name[:-2])

            for j in range(1, 1 + n_std_vars):
              if (j - 1) not in co_so:
                var_name = bien_names[j-1]
                cost = B[0, j]

                if abs(cost) < 1e-10:
                  col_vals = B[1:, j]
                  if any(c > 1e-10 for c in col_vals):
                    has_multiple_solutions = True
                    non_basic_vars_with_zero_cost.append(var_name)

                else:
                  if var_name.endswith('^+') or var_name.endswith('^-'):
                      base_name = var_name[:-2]
                      # Kiểm tra cặp xk^+ và xk^-
                      try:
                        idx_plus = bien_names.index(base_name + '^+')
                        idx_minus = bien_names.index(base_name + '^-')
                      except ValueError:
                        # Nếu không tìm thấy cả 2 biến, bỏ qua
                        continue

                    # Kiểm tra cả 2 biến đều không thuộc cơ sở
                      if (idx_plus not in co_so) and (idx_minus not in co_so):
                        cost_plus = B[0, idx_plus + 1]
                        cost_minus = B[0, idx_minus + 1]
                        col_plus = B[1:, idx_plus + 1]
                        col_minus = B[1:, idx_minus + 1]

                        if abs(cost_plus) < 1e-10 and abs(cost_minus) < 1e-10 and all(c <= 1e-10 for c in col_plus) and all(c <= 1e-10 for c in col_minus):
                            has_multiple_solutions = True
                            if base_name not in non_basic_vars_with_zero_cost:
                                non_basic_vars_with_zero_cost.append(base_name)
            if has_multiple_solutions:
                print("\nBài toán có vô số nghiệm tối ưu.")
                print("Kết luận nghiệm và giá trị tối ưu bài toán gốc (P):")

                if len(non_basic_vars_with_zero_cost) >= 1:
                    # Lấy biến tự do đầu tiên để biểu diễn các biến khác theo
                    free_var_name = non_basic_vars_with_zero_cost[0]
                    # Tìm chỉ số cột của biến tự do này trong bảng B
                    free_var_col_B_index = bien_names.index(free_var_name) + 1
                    # Tính hệ số tự do và hệ số theo biến tự do của từng biến trong nghiệm hiện tại
                    const_term_std = np.zeros(n_std_vars)
                    coeff_term_std = np.zeros(n_std_vars)

                    for j in range(n_std_vars):
                      if j in co_so:
                        row_idx = co_so.index(j) + 1  # Dòng trong B chứa biến này
                        const_term_std[j] = B[row_idx, 0]
                        coeff_term_std[j] = B[row_idx, free_var_col_B_index]
                      else:
                        const_term_std[j] = 0.0
                        coeff_term_std[j] = 0.0

                    lower_bound = 0.0
                    upper_bound = float('inf')

                    for i in range(1, m_std + 1):
                      basic_var = co_so[i - 1]
                      a_i = B[i, 0]  # Hệ số tự do (RHS)
                      b_i = B[i, free_var_col_B_index]  # hệ số của biến tự do trong hàng i

                      if abs(b_i) > 1e-8:
                        ratio = a_i / b_i
                        if b_i > 0:
                          upper_bound = min(upper_bound, ratio)
                        else:
                          lower_bound = max(lower_bound, ratio)


                      # Đếm chỉ số biến trong bài chuẩn
                      # Giả sử bạn đã có:
                      # const_term_std: vector hằng số trong nghiệm chuẩn (constant terms)
                      # coeff_term_std: vector hệ số biến tự do t trong nghiệm tổng quát (coefficient terms)
                      # co_so: tập chỉ số các biến cơ sở (basic variables)
                      # free_var_name = "t"  # tên biến tự do
                      # upper_bound, lower_bound: biên giới biến tự do (nếu có)
                      # val_primal: giá trị tối ưu

                      final_solution_strings = []
                      current_std_idx_for_recon = 0

                      for k in range(n_original_vars):
                        original_x_name = f"x{k+1}"
                        var_type = var_types[k]

                        if var_type == 'free':
                          # Giả sử biến tự do biến thành x+ và x-
                          y_plus_idx = current_std_idx_for_recon
                          y_minus_idx = current_std_idx_for_recon + 1
                          current_std_idx_for_recon += 2

                          const_part = const_term_std[y_plus_idx] - const_term_std[y_minus_idx]
                          coeff_part = coeff_term_std[y_plus_idx] - coeff_term_std[y_minus_idx]

                          is_current_x_basic = y_plus_idx in co_so or y_minus_idx in co_so

                          # Biểu diễn nghiệm tổng quát
                          if abs(coeff_part) < 1e-8:
                              expr = f"{original_x_name} = {const_part:.4f}"
                          else:
                              sign = "+" if coeff_part >= 0 else "-"
                              expr = f"{original_x_name} = {const_part:.4f} {sign} {abs(coeff_part):.4f} * {free_var_name}"

                        elif var_type == '>=':
                        # Biến <= 0, chuyển thành y = -x >= 0
                          y_idx = current_std_idx_for_recon
                          current_std_idx_for_recon += 1

                          is_current_x_basic = y_idx in co_so

                          const_part = -const_term_std[y_idx]  # đảo dấu
                          coeff_part = -coeff_term_std[y_idx]

                          if abs(coeff_part) < 1e-8:
                              expr = f"{original_x_name} = {const_part:.4f}"
                          else:
                              sign = "+" if coeff_part > 0 else "-"
                              expr = f"{original_x_name} = {const_part:.4f} {sign} {abs(coeff_part):.4f} * {free_var_name}"

                        elif var_type == '<=':
                        # Biến >= 0, giữ nguyên
                          y_idx = current_std_idx_for_recon
                          current_std_idx_for_recon += 1

                          is_current_x_basic = y_idx in co_so

                          # Đảo dấu khi tái tạo nghiệm nguyên thủy x = -y
                          const_part = const_term_std[y_idx]
                          coeff_part = coeff_term_std[y_idx]

                          if abs(coeff_part) < 1e-8:
                              expr = f"{original_x_name} = {const_part:.4f}"
                          else:
                              sign = "+" if coeff_part >= 0 else "-"
                              expr = f"{original_x_name} = {const_part:.4f} {sign} {abs(coeff_part):.4f} * {free_var_name}"

                        # Tạo biểu thức nghiệm tổng quát: x = a + b*t
                        if abs(coeff_part) < 1e-8:
                              expr = f"{original_x_name} = {const_part:.4f}"
                        else:
                              sign = "+" if coeff_part >= 0 else "-"
                              expr = f"{original_x_name} = {const_part:.4f} {sign} {abs(coeff_part):.4f} * {free_var_name}"

                        # Tái tạo x_k trong nghiệm tổng quát (với biến cơ sở và biến tự do)
                        if is_current_x_basic:
                          expr_str = f"{original_x_name} = {const_part:.4f}"
                          if abs(coeff_part) > 1e-10:
                            if coeff_part > 0:
                              expr_str += f" + {abs(coeff_part):.4f} * {free_var_name}"
                            else:
                              expr_str += f" - {abs(coeff_part):.4f} * {free_var_name}"
                          final_solution_strings.append(expr_str)
                        else:
                        # Biến không phải cơ sở, thường giá trị = 0
                          final_solution_strings.append(f"{original_x_name} = 0.0000")
                    print("Các biến cơ sở được biểu diễn theo biến tự do:")
                    for sol_str in final_solution_strings:
                      print(f"    {sol_str}")

                      if upper_bound != float('inf'):
                        print(f"Với {free_var_name} là biến tự do, {lower_bound:.4f} <= {free_var_name} <= {upper_bound:.4f}")
                      else:
                        print(f"Với {free_var_name} là biến tự do và {free_var_name} >= {lower_bound:.4f} (không giới hạn trên)")

                    print(f"Giá trị tối ưu của (P): {val_primal:.4f}")

                    return None, val_primal


            else: # Nghiệm tối ưu duy nhất
                print("\nBài toán có nghiệm tối ưu duy nhất.")
                print("\nKết luận nghiệm và giá trị tối ưu bài toán gốc (P):")
                x_opt_std_vars = np.zeros(n_std_vars)
                for i, var_idx in enumerate(co_so):
                    if var_idx < n_std_vars:
                        x_opt_std_vars[var_idx] = B[i+1, 0]

                final_x_values = np.zeros(n_original_vars)
                current_std_idx = 0
                for i in range(n_original_vars):
                    if var_types[i] == 'free':
                        final_x_values[i] = x_opt_std_vars[current_std_idx] - x_opt_std_vars[current_std_idx + 1]
                        current_std_idx += 2
                    elif var_types[i] == '<=':
                        final_x_values[i] = -x_opt_std_vars[current_std_idx]
                        current_std_idx += 1
                    else: # >=
                        final_x_values[i] = x_opt_std_vars[current_std_idx]
                        current_std_idx += 1

                for i in range(n_original_vars):
                    print(f"x{i+1} = {final_x_values[i]:.4f}")
                print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                return final_x_values, val_primal

        # In bảng từ vựng hiện tại trước khi pivot
        in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_std_vars, is_optimal_tableau=False)

        # Bước 4: Chọn biến ra (Quy tắc tỉ lệ tối thiểu: chọn hàng có tỉ lệ dương nhỏ nhất)

        bien_ra_row_idx = -1
        min_ratio = float('inf')

        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10: # Chỉ xét các hệ số dương để đảm bảo biến có thể giảm giá trị
                ratio = B[i, 0] / a_ij
                if ratio < min_ratio - 1e-10: # Nếu tỉ lệ nhỏ hơn
                    min_ratio = ratio
                    bien_ra_row_idx = i

        # Bước 5: Kiểm tra bài toán không giới nội
        if bien_ra_row_idx == -1:
            print("\nKhông tồn tại biến ra cho biến vào. Tất cả hệ số trong cột của biến vào đều <= 0.")
            print("=> Bài toán không giới nội (Unbounded).")
            if loai_bt == 'min':
                print("Giá trị tối ưu của bài toán gốc (P) là -∞ (âm vô cùng).")
            else:
                print("Giá trị tối ưu của bài toán gốc (P) là +∞ (dương vô cùng).")
            return None, None

        bien_co_so_cu_idx = co_so[bien_ra_row_idx-1]
        print(f"\nChọn biến vào: {bien_names[bien_vao_col_idx-1]}, biến ra: {bien_names[bien_co_so_cu_idx]}")

        # Bước 6: Thực hiện phép pivot
        # Cập nhật biến cơ sở
        co_so[bien_ra_row_idx-1] = bien_vao_col_idx - 1 # Biến vào thay thế biến ra

        # Chia hàng biến ra cho phần tử pivot
        pivot_element = B[bien_ra_row_idx, bien_vao_col_idx]
        B[bien_ra_row_idx, :] = B[bien_ra_row_idx, :] / pivot_element

        # Biến đổi các hàng khác
        for i in range(B.shape[0]):
            if i != bien_ra_row_idx:
                B[i, :] -= B[i, bien_vao_col_idx] * B[bien_ra_row_idx, :]

# Phuong phap Bland
def bland(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names):
    """
    Giải bài toán quy hoạch tuyến tính bằng phương pháp Bland để tránh vòng lặp.
    Hàm này thực hiện các bước pivot cho đến khi tìm được nghiệm tối ưu hoặc kết luận bài toán không giới nội.
    """
    m_std, n_std_vars = A_std.shape # m_std là số ràng buộc sau khi chuyển về dạng chuẩn, n_std_vars là số biến sau khi chuyển đổi (x', x'' và x)

    # Khởi tạo bảng từ vựng và danh sách biến cơ sở
    B, co_so = khoi_tao_bang_tu_vung(c_std, A_std, b_std)

    # bien_names bây giờ chính là standardized_var_names đã được truyền vào
    bien_names = standardized_var_names + [f"w{i+1}" for i in range(m_std)]

    pivot_count = 0 # Đếm số lần pivot

    final_x_values = None
    val_primal = None

    while True:
        pivot_count += 1
        bien_vao_col_idx = -1 # Chỉ số cột của biến vào (trong bảng B)

        # Bước 1: Chọn biến vào (theo Bland: hệ số âm đầu tiên)
        for j in range(1, 1 + n_std_vars + m_std): # Duyệt qua các cột biến (x và w)
            if B[0, j] < -1e-10:
                bien_vao_col_idx = j
                break

        # Bước 2: Kiểm tra điều kiện dừng (tối ưu)
        if bien_vao_col_idx == -1:
            # Nếu không tìm thấy hệ số âm nào trong hàng mục tiêu, bài toán đã tối ưu
            in_bang_tu_vung(B, bien_names, co_so, -1, pivot_count, n_std_vars, is_optimal_tableau=True) # In bảng tối ưu

            # Giá trị tối ưu của hàm mục tiêu (Z)
            z_opt_std = - B[0, 0] # Giá trị Z trong bảng từ vựng
            val_primal = -z_opt_std if loai_bt == 'max' else z_opt_std

            # Bước 3: Kiểm tra vô số nghiệm tối ưu
            has_multiple_solutions = False
            non_basic_vars_with_zero_cost = []

            for j in range(1, 1 + n_std_vars + m_std):
                # Kiểm tra xem biến có phải là biến không cơ sở không
                # co_so chứa chỉ số 0-indexed của biến trong bien_names
                # j-1 là chỉ số 0-indexed của biến trong bien_names tương ứng với cột j trong bảng B
                if (j - 1) not in co_so:
                    # Nếu là biến không cơ sở và có hệ số trong hàng mục tiêu xấp xỉ 0
                    if abs(B[0, j]) < 1e-10:
                        has_multiple_solutions = True
                        non_basic_vars_with_zero_cost.append(bien_names[j-1]) # Thêm tên biến vào danh sách

            if has_multiple_solutions:
                print("\nBài toán có vô số nghiệm tối ưu.")
                print("Kết luận nghiệm và giá trị tối ưu bài toán gốc (P):")

                if len(non_basic_vars_with_zero_cost) >= 1:
                    free_var_name = non_basic_vars_with_zero_cost[0]
                    free_var_col_B_index = bien_names.index(free_var_name) + 1

                    upper_bound = float('inf')
                    for i in range(1, m_std + 1):
                        coeff_in_row = B[i, free_var_col_B_index]
                        if coeff_in_row > 1e-10:
                            ratio = B[i, 0] / coeff_in_row
                            upper_bound = min(upper_bound, ratio)

                    lower_bound = 0.0

                    final_solution_strings = []
                    current_std_idx_for_recon = 0

                    for k in range(n_original_vars):
                        original_x_name = f"x{k+1}"

                        # Lấy các chỉ số biến chuẩn hóa tương ứng với biến gốc x_k
                        std_var_indices_for_xk = []
                        if var_types[k] == 'free':
                            std_var_indices_for_xk = [current_std_idx_for_recon, current_std_idx_for_recon + 1]
                            current_std_idx_for_recon += 2
                        else: # >= 0 hoặc <= 0
                            std_var_indices_for_xk = [current_std_idx_for_recon]
                            current_std_idx_for_recon += 1

                        # Kiểm tra xem biến gốc (hoặc một phần của nó) có phải là biến tự do được chọn không
                        is_this_the_free_var_relevance = False
                        for idx in std_var_indices_for_xk:
                            if idx < len(standardized_var_names) and standardized_var_names[idx] == free_var_name:
                                is_this_the_free_var_relevance = True
                                break

                        if is_this_the_free_var_relevance:
                            final_solution_strings.append(f"{original_x_name} (biến tự do)")
                            continue # Chuyển sang biến gốc tiếp theo

                        # Logic cho các biến gốc khác (KHÔNG phải biến tự do)
                        const_term = 0.0
                        coeff_term = 0.0
                        is_current_x_basic = False

                        if len(std_var_indices_for_xk) == 1: # x_k = y_k hoặc x_k = -y_k
                            std_var_idx = std_var_indices_for_xk[0]
                            if std_var_idx in co_so: # Nếu biến chuẩn hóa này là biến cơ sở
                                is_current_x_basic = True
                                row_in_basis = co_so.index(std_var_idx) + 1
                                const_term = B[row_in_basis, 0]
                                coeff_term = -B[row_in_basis, free_var_col_B_index]

                            if var_types[k] == '<=': # x_k = -y_k, đảo dấu kết quả
                                const_term = -const_term
                                coeff_term = -coeff_term
                        elif len(std_var_indices_for_xk) == 2: # x_k = x_k^+ - x_k^- (biến tự do ban đầu)
                            x_plus_idx = std_var_indices_for_xk[0]
                            x_minus_idx = std_var_indices_for_xk[1]

                            # Kiểm tra xem ít nhất một trong x_k^+ hoặc x_k^- có phải là biến cơ sở không
                            if x_plus_idx in co_so or x_minus_idx in co_so:
                                is_current_x_basic = True

                                val_x_plus = 0.0
                                coeff_x_plus_free_var = 0.0
                                if x_plus_idx in co_so:
                                    row_in_basis_plus = co_so.index(x_plus_idx) + 1
                                    val_x_plus = B[row_in_basis_plus, 0]
                                    coeff_x_plus_free_var = B[row_in_basis_plus, free_var_col_B_index]

                                val_x_minus = 0.0
                                coeff_x_minus_free_var = 0.0
                                if x_minus_idx in co_so:
                                    row_in_basis_minus = co_so.index(x_minus_idx) + 1
                                    val_x_minus = B[row_in_basis_minus, 0]
                                    coeff_x_minus_free_var = B[row_in_basis_minus, free_var_col_B_index]

                                const_term = val_x_plus - val_x_minus
                                coeff_term = -(coeff_x_plus_free_var - coeff_x_minus_free_var)

                        if is_current_x_basic:
                            expr_str = f"{original_x_name} = {const_term:.4f}"
                            if abs(coeff_term) > 1e-10:
                                if coeff_term > 0:
                                    expr_str += f" + {abs(coeff_term):.4f} * {free_var_name}"
                                else:
                                    expr_str += f" - {abs(coeff_term):.4f} * {free_var_name}"
                            final_solution_strings.append(expr_str)
                        else: # Biến không cơ sở (không phải biến tự do được chọn), nên giá trị bằng 0
                            final_solution_strings.append(f"{original_x_name} = 0.0000")

                    print("Các biến cơ sở được biểu diễn theo biến tự do:")
                    for sol_str in final_solution_strings:
                        print(f"    {sol_str}")

                    if upper_bound != float('inf'):
                        print(f"Với {free_var_name} là biến tự do, {lower_bound:.4f} <= {free_var_name} <= {upper_bound:.4f}")
                    else:
                        print(f"Với {free_var_name} là biến tự do và {free_var_name} >= {lower_bound:.4f} (không giới hạn trên)")
                    print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                    return None, val_primal

            else: # Nghiệm tối ưu duy nhất
                print("\nBài toán có nghiệm tối ưu duy nhất.")
                print("\nKết luận nghiệm và giá trị tối ưu bài toán gốc (P):")
                x_opt_std_vars = np.zeros(n_std_vars)
                for i, var_idx in enumerate(co_so):
                    if var_idx < n_std_vars:
                        x_opt_std_vars[var_idx] = B[i+1, 0]

                final_x_values = np.zeros(n_original_vars)
                current_std_idx = 0
                for i in range(n_original_vars):
                    if var_types[i] == 'free':
                        final_x_values[i] = x_opt_std_vars[current_std_idx] - x_opt_std_vars[current_std_idx + 1]
                        current_std_idx += 2
                    elif var_types[i] == '<=':
                        final_x_values[i] = -x_opt_std_vars[current_std_idx]
                        current_std_idx += 1
                    else: # >= 0
                        final_x_values[i] = x_opt_std_vars[current_std_idx]
                        current_std_idx += 1

                for i in range(n_original_vars):
                    print(f"x{i+1} = {final_x_values[i]:.4f}")
                print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                return final_x_values, val_primal

        in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_std_vars, is_optimal_tableau=False)

        # Bước 4: Chọn biến ra (Quy tắc Bland: chọn hàng có tỉ lệ nhỏ nhất,
        # Nếu bằng nhau thì ưu tiên biến cơ sở có chỉ số nhỏ nhất)

        bien_ra_row_idx = -1
        min_ratio = float('inf')

        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10:
                ratio = B[i, 0] / a_ij

                if ratio < min_ratio - 1e-10 or \
                   (abs(ratio - min_ratio) < 1e-10 and \
                    (bien_ra_row_idx == -1 or co_so[i-1] < co_so[bien_ra_row_idx-1])):
                    min_ratio = ratio
                    bien_ra_row_idx = i

        if bien_ra_row_idx == -1:
            print("Không tồn tại biến ra cho biến vào => bài toán không giới nội.")
            if loai_bt == 'min':
                print("Giá trị tối ưu của bài toán gốc (P) là -oo (âm vô cùng).")
            else:
                print("Giá trị tối ưu của bài toán gốc (P) là +oo (dương vô cùng).")
            return None, None

        bien_co_so_cu_idx = co_so[bien_ra_row_idx-1]
        print(f"Biến vào: {bien_names[bien_vao_col_idx-1]}, biến ra: {bien_names[bien_co_so_cu_idx]}")

        co_so[bien_ra_row_idx-1] = bien_vao_col_idx - 1

        pivot_element = B[bien_ra_row_idx, bien_vao_col_idx]
        B[bien_ra_row_idx, :] = B[bien_ra_row_idx, :] / pivot_element

        for i in range(B.shape[0]):
            if i != bien_ra_row_idx:
                B[i, :] -= B[i, bien_vao_col_idx] * B[bien_ra_row_idx, :]

        in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_std_vars, is_optimal_tableau=False)

        # Bước 4: Chọn biến ra (Quy tắc Bland: chọn hàng có tỉ lệ nhỏ nhất,
        # Nếu bằng nhau thì ưu tiên biến cơ sở có chỉ số nhỏ nhất)

        bien_ra_row_idx = -1
        min_ratio = float('inf')

        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10:
                ratio = B[i, 0] / a_ij

                if ratio < min_ratio - 1e-10 or \
                   (abs(ratio - min_ratio) < 1e-10 and \
                    (bien_ra_row_idx == -1 or co_so[i-1] < co_so[bien_ra_row_idx-1])):
                    min_ratio = ratio
                    bien_ra_row_idx = i

        if bien_ra_row_idx == -1:
            print("Không tồn tại biến ra cho biến vào => bài toán không giới nội.")
            if loai_bt == 'min':
                print("Giá trị tối ưu của bài toán gốc (P) là -oo (âm vô cùng).")
            else:
                print("Giá trị tối ưu của bài toán gốc (P) là +oo (dương vô cùng).")
            return None, None

        bien_co_so_cu_idx = co_so[bien_ra_row_idx-1]
        print(f"Biến vào: {bien_names[bien_vao_col_idx-1]}, biến ra: {bien_names[bien_co_so_cu_idx]}")

        co_so[bien_ra_row_idx-1] = bien_vao_col_idx - 1

        pivot_element = B[bien_ra_row_idx, bien_vao_col_idx]
        B[bien_ra_row_idx, :] = B[bien_ra_row_idx, :] / pivot_element

        for i in range(B.shape[0]):
            if i != bien_ra_row_idx:
                B[i, :] -= B[i, bien_vao_col_idx] * B[bien_ra_row_idx, :]


# Phuong phap hai pha
def hai_pha(A, b, c, loai):



    def print_tableau(tableau, step=None):
        print("\n========== Bảng từ vựng {} ==========".format(step if step is not None else ""))
        if isinstance(tableau, np.ndarray):
            print(np.round(tableau, 4))
        else:
            print(tableau)

    def pivot_step(tableau, row, col):
        pivot = tableau[row, col]
        tableau[row, :] = tableau[row, :] / pivot
        for i in range(len(tableau)):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]
        tableau[row, :] *= -1
        return tableau

    def pha1(tableau, x0_col):

        def don_hinh_pha1(tableau, x0_col, start_step=1):
            step = start_step

            while True:
                cj = tableau[-1, :-1]
                is_done = np.isclose(cj[x0_col], 1.0) and np.all(np.isclose(np.delete(cj, x0_col), 0))
                if is_done:
                    print("\n Dừng: Hàm mục tiêu Z = x0.")
                    print_tableau(tableau, f"kết thúc pha 1")
                    return tableau

                cj_temp = cj.copy()
                candidates = np.where(cj_temp < 0)[0]

                if len(candidates) == 0:
                    print("\n Không còn cột cj âm để chọn → bài toán vô nghiệm.")
                    return None

                col = candidates[np.argmin(cj_temp[candidates])]

                col_vals = tableau[:-1, col]
                b_vals = tableau[:-1, -1]

                valid_rows = [i for i in range(len(col_vals)) if col_vals[i] < 0]
                if not valid_rows:
                    print("\n Tập chấp nhận rỗng → bài toán vô nghiệm trong pha 1.")
                    return None

                ratios = [b_vals[i] / abs(col_vals[i]) for i in valid_rows]
                row = valid_rows[np.argmin(ratios)]

                print(f"\n Pivot tại row = {row}, col = {col} (đơn hình pha 1)")
                tableau = pivot_step(tableau, row, col)
                step += 1
                print_tableau(tableau, step)

        print_tableau(tableau, "ban đầu")
        b_vals = tableau[:-1, -1]
        row = np.argmin(b_vals)
        print(f"\nPivot đầu tiên: vào x0 (col={x0_col}), ra row={row}")
        tableau = pivot_step(tableau, row, x0_col)
        print_tableau(tableau, "sau pivot x0")

        tableau = don_hinh_pha1(tableau, x0_col, start_step=1)
        return tableau

    def pha2(tableau, c, A, b):

        def tao_lai_z_pha2(tableau, c, x0_col):
            tableau = np.delete(tableau, x0_col, axis=1)
            m_tab, n_tab = tableau.shape
            m = m_tab - 1
            n = len(c)

            z_moi = np.zeros(n_tab)

            for j in range(n):
                col = tableau[:-1, j]
                if np.count_nonzero(col) == 1 and np.sum(col) == -1:
                    i = np.where(col == -1)[0][0]
                    c_j = c[j]

                    z_moi[j] = 0
                    for k in range(n_tab - 1):
                        if k != j:
                            z_moi[k] += c_j * tableau[i, k]
                    z_moi[-1] += c_j * tableau[i, -1]

            for j in range(n):
                col = tableau[:-1, j]
                if not (np.count_nonzero(col) == 1 and np.sum(col) == -1):
                    z_moi[j] += c[j]

            tableau[-1, :] = z_moi

            return tableau

        def don_hinh_pha2(tableau, start_step=1):
            step = start_step
            while True:
                cj = tableau[-1, :-1]
                if np.all(cj >= 0):
                    print("\nĐạt nghiệm tối ưu (pha 2) ")
                    infinite = False
                    n = len(cj)
                    for j in range(n):
                        col = tableau[:-1, j]
                        is_basic = (np.count_nonzero(col) == 1 and np.sum(col) == -1)
                        if not is_basic and np.isclose(cj[j], 0):
                            infinite = True
                            break
                    if infinite:
                        return "INFINITE_SOLUTIONS", tableau

                    return "optimal", tableau

                cj_temp = cj.copy()
                candidate = np.where(cj_temp < 0)[0]
                if len(candidate) == 0:
                    print("\n Không tìm được biến vào → đạt nghiệm tối ưu.")
                    return "optimal", tableau
                col = candidate[np.argmin(cj_temp[candidate])]
                col_vals = tableau[:-1, col]
                b_vals = tableau[:-1, -1]

                valid_rows = [i for i in range(len(col_vals)) if col_vals[i] < 0]
                if not valid_rows:
                    print("\n không tìm được biến ra  => Bài toán không giới nội")
                    return "UNBOUNDED", tableau

                ratios = [b_vals[i] / abs(col_vals[i]) for i in valid_rows]
                row = valid_rows[np.argmin(ratios)]

                print(f"\n Pivot tại row = {row}, col = {col} (đơn hình pha 2)")
                tableau = pivot_step(tableau, row, col)
                step += 1
                print_tableau(tableau, step)

        tableau = tao_lai_z_pha2(tableau, c, x0_col=A.shape[1])
        print_tableau(tableau, "Cho x0 = 0")
        result, final_tab = don_hinh_pha2(tableau, start_step=1)
        return result, final_tab


    def create_initial_tableau(A, b):
        m, n = A.shape
        A = np.array(A)
        b = np.array(b)

        neg_A = -1 * A
        x0_col = np.ones((m, 1))
        I = -1 * np.eye(m)
        b_col = b.reshape(-1, 1)
        tableau = np.hstack((neg_A, x0_col, I, b_col))

        total_cols = n + 1 + m + 1
        z = np.zeros((1, total_cols))
        z[0, n] = 1
        tableau = np.vstack((tableau, z))
        return tableau

    def in_ket_qua_cuoi_cung(result, tableau, c, A, loai):
        if tableau is None:
            print("\n Bài toán vô nghiệm.")
            return

        if result == "UNBOUNDED":
            if loai == 'max':
                print("\n Bài toán không bị chặn trên ⇒ Z = +∞")
            else:
                print("\n Bài toán không bị chặn dưới ⇒ Z = -∞")
        elif result == "INFINITE_SOLUTIONS":
            print("\nBài toán có vô số nghiệm tối ưu.")
            print("\nKết quả cuối cùng:")
            m, n_plus_m_plus_1 = tableau.shape
            m -= 1
            n = len(c)

            z = tableau[-1, -1]
            if loai == 'max':
                z *= -1

            print(f"\n Giá trị tối ưu của hàm mục tiêu Z = {z:.4f}")
            # Không in nghiệm tối ưu
        elif result == "optimal":
            print("\nBài toán có nghiệm tối ưu duy nhất.")
            print("\nKết quả cuối cùng:")
            m, n_plus_m_plus_1 = tableau.shape
            m -= 1
            n = len(c)

            z = tableau[-1, -1]
            if loai == 'max':
                z *= -1

            print(f"\n Giá trị tối ưu của hàm mục tiêu Z = {z:.4f}")
            x_vals = np.zeros(n)
            for j in range(n):
                col = tableau[:-1, j]
                if np.count_nonzero(col) == 1 and np.sum(col) == -1:
                    i = np.where(col == -1)[0][0]
                    x_vals[j] = tableau[i, -1]

            for i, val in enumerate(x_vals):
                print(f"x{i+1} = {val:.4f}")

    tableau = create_initial_tableau(A, b)
    print_tableau(tableau)
    tableau = pha1(tableau, x0_col=A.shape[1])
    if tableau is not None:
        result, tableau = pha2(tableau, c, A, b)
        print_tableau(tableau)
        in_ket_qua_cuoi_cung(result, tableau, c, A, loai)

    else:
        in_ket_qua_cuoi_cung(None, None, c, A, loai)


# Ham main
if __name__ == "__main__":
    loai_bt, c_original, A_original, b_original, rls, n_original_vars, var_types = nhap_bai_toan()

    # Chuyển bài toán về dạng chuẩn:
    c_std, A_std, b_std, n_std_vars, standardized_var_names = chuyen_ve_dang_chuan(loai_bt, c_original, A_original, b_original, rls, var_types)

    # Gọi hàm để xác định và cho phép người dùng chọn phương pháp
    phuong_phap_duoc_chon = xet_phuong_phap(n_original_vars, b_std)

    if phuong_phap_duoc_chon == 1:
        print("\n--- Bạn đã chọn: Giải bằng phương pháp hình học ---")
        final_x_values, val_primal = hinh_hoc(A_original, b_original, c_original, loai_bt, rls, var_types, standardized_var_names)
    elif phuong_phap_duoc_chon == 2:
        print("\n--- Bạn đã chọn: Giải bằng phương pháp đơn hình ---")
        final_x_values, val_primal= don_hinh(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names)
    elif phuong_phap_duoc_chon == 3:
        print("\n--- Bạn đã chọn: Giải bằng phương pháp Bland ---")
        final_x_values, val_primal = bland(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names)
    else: # phuong_phap_duoc_chon == 4
        print("\n--- Bạn đã chọn: Giải bằng phương pháp hai pha ---")
        hai_pha(A_std,b_std,c_std,loai_bt)
        # Gọi hàm giải bằng phương hai pha ở đây