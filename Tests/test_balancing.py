import unittest
from Tests.BaseTest import *
from Tests.errors import get_error, get_error_txt
from Tools.DatasetBalanced import DatasetBalanced

FILE_DATASET = 'Datasets/Tests/clasificacion-sintetico-desbalanceado_v1.csv'
BALANCE_METHODS = list(DatasetBalanced.METHODS.keys())

class TestNormalization(BaseTest):

    def test_normalization(self):
        io_data = self.get_iodata()
        xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples = self.get_dataset(io_data)
        args = Args('DT')
        x_init_shape = xts.shape
        y_init_shape = yts.shape

        for m in BALANCE_METHODS:
            if m == 'ADASYN': # exclude it because it takes too long
                continue

            args.balanced = [m]
            xts, yts, idx_samples = DatasetBalanced().choice_method_balanced(xts, yts, args, idx_samples)

            self.assertTrue(x_init_shape[0] <= xts.shape[0], get_error_txt(ERROR_BALANCING, m))
            self.assertTrue(y_init_shape[0] <= yts.shape[0], get_error_txt(ERROR_BALANCING, m))
            self.assertTrue(xts.shape[0] == yts.shape[0], get_error_txt(ERROR_BALANCING, m))

if __name__ == '__main__':
    unittest.main()
