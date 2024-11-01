import numpy as np
import unittest
from os.path import basename
from Tests.errors import *
from Tests.BaseTest import *
from Tools.ToolsModels import is_tf_model
from Common.Analysis.EvaluationMetrics import EvaluationMetrics, TypeML
from Models.BaseModel import BaseModel

DATASET_LEN = 100
DATASET_LEN_IDLIST = 4

class TestPrediction(BaseTest):

    def test_prediction(self):
        lst_models = glob(FOLDER_MODELS_TEST + '*')
        self.assertTrue(len(lst_models) == NUMBER_MODELS, get_error_txt(ERROR_N_MODELS, FOLDER_MODELS_TEST))

        io_data = self.get_iodata()
        xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples = self.get_dataset(io_data)
        X = np.concatenate((xtr, xts))
        Y = np.concatenate((ytr, yts))

        for file_model in lst_models:
            model = BaseModel.load(file_model)
            name_model = basename(file_model).split("_")[0]

            params = {'type_ml': TypeML.CLASSIFICATION.value, 'model': name_model, 'classification_type': 'binary'}
            p = FOLDER_TEST + name_model
            args = Args(name_model)
            cfg = self.get_config_holder(args, params, p)

            ypr = np.squeeze(model.predict(X))

            if not args.regression:
                if is_tf_model(model):
                    ypr = ypr.argmax(axis=1)
                else:
                    ypr = ypr.astype(int)

            self.assertEquals(len(np.unique(ypr)), 2, get_error_txt(ERROR_NOT_BINARY, name_model))
            self.assertEquals(len(ypr), len(Y), get_error_txt(ERROR_DIFF_LENGTH, name_model))

            EvaluationMetrics(Y, ypr, X, cfg, model, id_list, io_data).all_metrics()

            self.assertEqual(self.count_files_by_pattern(name_model + '*confusion_matrix.png'), 1, get_error(ERROR_CONF_MATRIX))
            self.assertEqual(self.count_files_by_pattern(name_model + '*roc_proba_class.png'), 1, get_error(ERROR_ROC_PROBA_CLASS))
            self.assertEqual(self.count_files_by_pattern(name_model + '*data.json'), 1, get_error(ERROR_DATA_JSON))
            self.assertEqual(self.count_files_by_pattern(name_model + '*resume.txt'), 1, get_error(ERROR_DATA_TXT))
            self.assertTrue(self.count_files_by_pattern(name_model + '*roc_proba_*.png') > 0, get_error(ERROR_ROC_PROBA))
            self.assertTrue(self.count_files_by_pattern(name_model + '*roc_*.png') > 0, get_error(ERROR_ROC))

            print('{:<8}{}{:>3}{} '.format(name_model, bcolors.OKGREEN, "Ok", bcolors.ENDC))

if __name__ == "__main__":
    unittest.main()
