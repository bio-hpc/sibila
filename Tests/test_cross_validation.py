import unittest
from Common.Config.config import get_default_config
from Models import *
from Models.Utils.CrossValidation import CrossValidation
from Tests.BaseTest import *
from Tests.errors import get_error, get_error_txt

CV_METHODS = list(CrossValidation.METHODS.keys())

class TestCrossValidation(BaseTest):

    def test_cross_validation(self):
        model_name = 'ANN'
        io_data = self.get_iodata()
        xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples = self.get_dataset(io_data)

        args = Args(model_name)
        params = get_default_config(model_name, io_data)

        params['params']['cv_splits'] = 5
        params['params_grid']['epochs'] = 3
        params['params_grid']['min_units'] = 2
        params['params_grid']['max_units'] = 8
        params['params_grid']['max_layers'] = 2
        params['params_grid']['executions_per_trial'] = 2
        params['params_grid']['min_lr'] = 0.01
        params['params_grid']['max_lr'] = 0.01
        p = FOLDER_TEST + model_name
        cfg = self.get_config_holder(args, params, p)
        model = globals()[model_name](io_data, cfg, id_list)

        for m in CV_METHODS[:1]: # FIXME run all the methods
            args.crossvalidation = m
            model.train(xtr, ytr)
            # there's no need to assert anything because we just want to prove it finishes correctly

if __name__ == '__main__':
    unittest.main()
