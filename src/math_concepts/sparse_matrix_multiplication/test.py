import unittest
import solution


class TestProgram(unittest.TestCase):
    def test_case_1(self):
        matrix_a = [
            [0, 2, 0],
            [0, -3, 5],
        ]
        matrix_b = [
            [0, 10, 0],
            [0, 0, 0],
            [0, 0, 4],
        ]
        expected = [
            [0, 0, 0],
            [0, 0, 20],
        ]
        actual = solution.sparse_matrix_multiplication(matrix_a, matrix_b)
        self.assertEqual(actual, expected)

    def test_case_2(self):
        matrix_a = [
            [46, 0, 0],
            [45, 47, 0],
            [0, 0, 0],
            [34, 0, 25],
            [0, 2, 0],
            [0, 0, 0],
        ]
        matrix_b = [
            [26, 34, 20, 31, 34, 15],
            [38, 30, 23, 1, 45, 22],
            [47, 9, 9, 5, 9, 31],
        ]
        expected = [
            [1196, 1564, 920, 1426, 1564, 690],
            [2956, 2940, 1981, 1442, 3645, 1709],
            [0, 0, 0, 0, 0, 0],
            [2059, 1381, 905, 1179, 1381, 1285],
            [76, 60, 46, 2, 90, 44],
            [0, 0, 0, 0, 0, 0],
        ]
        actual = solution.sparse_matrix_multiplication(matrix_a, matrix_b)
        self.assertEqual(actual, expected)

    def test_case_3(self):
        matrix_a = [
            [0, 0, 1],
            [1, 0, 2],
            [0, 0, 1],
        ]
        matrix_b = [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
        expected = [
            [0, 1, 0],
            [0, 3, 0],
            [0, 1, 0],
        ]
        actual = solution.sparse_matrix_multiplication(matrix_a, matrix_b)
        self.assertEqual(actual, expected)
