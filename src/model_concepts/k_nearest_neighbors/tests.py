import json

import unittest
from solution import find_k_nearest_neighbors, predict_label


def get_knn_examples():
    with open("./data.json", "rb") as handle:
        return json.load(handle)


EXAMPLES = get_knn_examples()


class TestProgram(unittest.TestCase):
    def test_case_1(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 1
        expected = ["pid_500"]
        actual = find_k_nearest_neighbors(EXAMPLES, features, k)
        self.assertEqual(sorted(actual), sorted(expected))

    def test_case_2(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 1
        expected = 1
        actual = predict_label(EXAMPLES, features, k)
        self.assertEqual(actual, expected)

    def test_case_3(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 3
        expected = ["pid_500", "pid_535", "pid_545"]
        actual = find_k_nearest_neighbors(EXAMPLES, features, k)
        self.assertEqual(sorted(actual), sorted(expected))

    def test_case_4(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 3
        expected = 1
        actual = predict_label(EXAMPLES, features, k)
        self.assertEqual(actual, expected)

    def test_case_5(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 5
        expected = ["pid_500", "pid_535", "pid_545", "pid_512", "pid_516"]
        actual = find_k_nearest_neighbors(EXAMPLES, features, k)
        self.assertEqual(sorted(actual), sorted(expected))

    def test_case_6(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 5
        expected = 1
        actual = predict_label(EXAMPLES, features, k)
        self.assertEqual(actual, expected)

    def test_case_7(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 7
        expected = [
            "pid_500",
            "pid_535",
            "pid_545",
            "pid_512",
            "pid_516",
            "pid_513",
            "pid_537",
        ]
        actual = find_k_nearest_neighbors(EXAMPLES, features, k)
        self.assertEqual(sorted(actual), sorted(expected))

    def test_case_8(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 7
        expected = 1
        actual = predict_label(EXAMPLES, features, k)
        self.assertEqual(actual, expected)

    def test_case_9(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 9
        expected = [
            "pid_500",
            "pid_535",
            "pid_545",
            "pid_512",
            "pid_516",
            "pid_513",
            "pid_537",
            "pid_540",
            "pid_528",
        ]
        actual = find_k_nearest_neighbors(EXAMPLES, features, k)
        self.assertEqual(sorted(actual), sorted(expected))

    def test_case_10(self):
        features = [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593]
        k = 9
        expected = 0
        actual = predict_label(EXAMPLES, features, k)
        self.assertEqual(actual, expected)
