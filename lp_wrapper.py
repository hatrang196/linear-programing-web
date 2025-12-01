from linear_programing_solver import (
    convert,
    hinh_hoc,
    standard_form,
    simplex_algorithm,
    two_phase_method,
    bland_method
)

def solve_lp(A, b, c, loai, rls, var_types, method="Simplex"):
    """
    method: "Geometric", "Simplex", "Two-phase", "Bland"
    """

    if method == "Geometric":
        final_x, z = hinh_hoc(A, b, c, loai, rls, var_types)
        return f"Nghiệm tối ưu: {final_x}\nGiá trị Z: {z}"

    if method == "Simplex":
        final_x, z = simplex_algorithm(A, b, c, loai, rls, var_types)
        return f"Nghiệm tối ưu: {final_x}\nGiá trị Z: {z}"

    if method == "Two-phase":
        final_x, z = two_phase_method(A, b, c, loai, rls, var_types)
        return f"Nghiệm tối ưu: {final_x}\nGiá trị Z: {z}"

    if method == "Bland":
        final_x, z = bland_method(A, b, c, loai, rls, var_types)
        return f"Nghiệm tối ưu: {final_x}\nGiá trị Z: {z}"

    return "Không nhận dạng được thuật toán."
