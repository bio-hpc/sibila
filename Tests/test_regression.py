import unittest
from Common.Analysis.EvaluationMetrics import EvaluationMetrics, TypeML
from Common.Analysis.MergeResults import MergeResults
from Common.Config.config import get_default_config
from Models import *
from Tests.BaseTest import *
from Tests.errors import get_error, get_error_txt


FILE_DATASET = 'Datasets/Tests/regresion-lineal_v1.csv'

MODEL_ASSERTS = {
    'ANN':[
        { '*loss_acc.png': 'E0111' }
    ],
    'DT':[
        { '*file.dot': 'E0109' },
        { '*file.png': 'E0110' }
    ]
}

class TestRegression(BaseTest):

    def test_regression(self):
        io_data = self.get_iodata()
        xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples = self.get_dataset(io_data)

        for m in io_data.read_all_options():
            # RP and RLF don't support regression
            # The default config of ANN is only valid for classification
            if m in ['RP','RLF','ANN']:
                continue

            args = Args(m)
            args.regression = True
            params = get_default_config(m, io_data)
            params['type_ml'] = TypeML.REGRESSION.value
            p = FOLDER_TEST + m
            cfg = self.get_config_holder(args, params, p)
            
            model = globals()[m](io_data, cfg, id_list)
            model.train(xtr, ytr)
            ypr = model.predict(xts)
            BaseModel.save_model(cfg, model.get_model())

            EvaluationMetrics(yts, ypr, xts, cfg, model.get_model(), id_list, io_data).all_metrics()
            MergeResults(args.folder)

            # common asserts
            n_joblib = self.count_files_by_pattern(m + '*.joblib')
            n_h5 = self.count_files_by_pattern(m + '*.h5')
            n_dat = self.count_files_by_pattern(m + '*.dat')

            self.assertEqual(n_joblib + n_h5 + n_dat, 1, get_error(ERROR_SAVED_MODEL))
            self.assertEqual(self.count_files_by_pattern(m + '*correlation.png'), 1, get_error(ERROR_CORRELATION))
            self.assertEqual(self.count_files_by_pattern(m + '*data.json'), 1, get_error(ERROR_DATA_JSON))
            self.assertEqual(self.count_files_by_pattern(m + '*resume.txt'), 1, get_error(ERROR_DATA_TXT))

            # model-dependant asserts
            if m in MODEL_ASSERTS:
                for a in MODEL_ASSERTS[m]:
                    for x in a.keys():
                        self.assertEqual(self.count_files_by_pattern(m + x), 1, get_error_txt(ERROR_MODEL, '{} - {}'.format(m, x)))            

        self.assertEqual(self.count_files_by_pattern('Resume_analysis.csv'), 1, get_error(ERROR_DATA_CSV))
            
if __name__ == '__main__':
    unittest.main()
