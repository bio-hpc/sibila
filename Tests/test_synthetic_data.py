import unittest
import numpy as np

from Tests.synthetic_data import make_classification_dataset, make_regression_dataset


class TestSyntheticData(unittest.TestCase):

    def test_classification_is_deterministic(self):
        first = make_classification_dataset(seed=2021)
        second = make_classification_dataset(seed=2021)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_regression_shape(self):
        x, y, id_list, idx_samples, target_classes = make_regression_dataset(seed=2021)
        self.assertEqual(x.shape, (71, 6))
        self.assertEqual(len(y), 71)
        self.assertEqual(len(id_list), 6)
        self.assertIsNone(target_classes)

    def test_imbalanced_classification(self):
        _, y, _, _, _ = make_classification_dataset(seed=2021, imbalanced=True)
        counts = np.bincount(y)
        self.assertEqual(counts.sum(), 100)
        self.assertGreater(max(counts), min(counts))


if __name__ == '__main__':
    unittest.main()
