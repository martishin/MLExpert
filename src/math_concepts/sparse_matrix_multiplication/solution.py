from typing import List, Tuple, Dict


def sparse_matrix_multiplication(
    matrix_a: List[List[int]], matrix_b: List[List[int]]
) -> List[List[int]]:
    if len(matrix_a[0]) != len(matrix_b):
        return [[]]

    sparse_a = get_dict_of_nonzero_cells(matrix_a)
    sparse_b = get_dict_of_nonzero_cells(matrix_b)

    matrix_c = [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))]

    for (row_a, col_a), value_a in sparse_a.items():
        for (row_b, col_b), value_b in sparse_b.items():
            if col_a == row_b:
                matrix_c[row_a][col_b] += value_a * value_b

    return matrix_c


def get_dict_of_nonzero_cells(matrix: List[List[int]]) -> Dict[Tuple[int, int], int]:
    dict_of_nonzero_cells = {}
    for row, row_values in enumerate(matrix):
        for col, value in enumerate(row_values):
            if value != 0:
                dict_of_nonzero_cells[(row, col)] = value
    return dict_of_nonzero_cells
