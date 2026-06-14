import unittest
import numpy as np
import os
import tempfile

from Tools.FeatureReduction import FeatureReduction
from Tools.IOData import IOData


class TestFeatureReduction(unittest.TestCase):

    def setUp(self):
        self.fr = FeatureReduction()
        self.tmp_dir = tempfile.mkdtemp()
        self.io_data = IOData()
        self.io_data.set_file_resume(os.path.join(self.tmp_dir, 'Experiment_out.txt'))
        open(self.io_data.get_file_resume(), 'w').close()

        np.random.seed(2020)
        self.x = np.random.randn(40, 8)
        self.x[:, 0] = np.nan
        self.x[:, 1] = 0
        self.y = np.array([0, 1] * 20)
        self.id_list = ['f{}'.format(i) for i in range(8)]

    def test_backward_selection_respects_max_features_cap(self):
        fr_cfg = {
            'fill_nan': 'mean',
            'selection': 'backward',
            'p_value_threshold': 0.05,
            'max_features': 5,
            'min_features': 1
        }
        x_out, id_out, y_out, state = self.fr.apply(
            self.x, self.y, self.id_list, fr_cfg, False, self.io_data
        )
        self.assertLessEqual(x_out.shape[1], 5)
        self.assertLessEqual(len(id_out), 5)
        self.assertEqual(len(y_out), 40)

    def test_p_value_selection_keeps_variable_feature_count(self):
        rng = np.random.RandomState(2020)
        x = rng.randn(80, 12)
        y = (x[:, 0] + x[:, 1] + rng.normal(0, 0.2, 80) > 0).astype(int)
        id_list = ['f{}'.format(i) for i in range(12)]

        fr_cfg = {
            'selection': 'backward',
            'p_value_threshold': 0.05,
            'max_features': None,
            'min_features': 1
        }
        x_out, id_out, _, _ = self.fr.apply(
            x, y, id_list, fr_cfg, False, self.io_data
        )
        self.assertGreaterEqual(len(id_out), 1)
        self.assertLess(len(id_out), 12)

    def test_outliers_cap_values_and_preserve_rows(self):
        x = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 5.0],
        ])
        y = np.array([0, 1, 0, 1])
        id_list = ['f0', 'f1']

        fr_cfg = {'outliers': 'turkey'}
        x_out, _, y_out, _ = self.fr.apply(x, y, id_list, fr_cfg, False, self.io_data)

        self.assertEqual(x_out.shape[0], 4)
        self.assertEqual(len(y_out), 4)
        self.assertLess(x_out[3, 0], 100.0)

    def test_save_and_load_state(self):
        prefix = os.path.join(self.tmp_dir, 'RF')
        kept = ['f2', 'f3', 'f4']
        removed = ['f0', 'f1']
        self.fr.save_state(prefix, kept, removed, self.io_data)

        x_pred = np.random.randn(3, 8)
        id_pred = self.id_list
        x_out, id_out = self.fr.apply_from_state(
            x_pred, id_pred, prefix + '.joblib', self.io_data
        )
        self.assertEqual(x_out.shape, (3, 3))
        self.assertEqual(id_out, kept)


if __name__ == '__main__':
    unittest.main()
