import unittest
from Tests.BaseTest import *
from Tests.errors import get_error, get_error_txt
from Tools.DataNormalization import DataNormalization

NORMALIZATION_METHODS = list(DataNormalization.METHODS.keys())

class TestNormalization(BaseTest):

    def test_normalization(self):
        io_data = self.get_iodata()
        xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples = self.get_dataset(io_data)
        args = Args('DT')
        expected_shape = xtr.shape

        for m in NORMALIZATION_METHODS:
            if m == 'BC':
                continue

            args.normalize = [m]
            xtr = DataNormalization().choice_method_normalize(xtr, args)
            if m in ['SMA','SMB','SMM','PF']:
                self.assertEqual(expected_shape[0], xtr.shape[0], get_error_txt(ERROR_NORMALIZATION, m))
            else:
                self.assertEqual(expected_shape, xtr.shape, get_error_txt(ERROR_NORMALIZATION, m))


if __name__ == '__main__':
    unittest.main()
