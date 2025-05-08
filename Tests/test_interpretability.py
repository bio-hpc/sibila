import unittest
import time
from os.path import basename, isdir, join, getsize
from glob import glob
import shutil
from Tests.errors import *
from Tests.BaseTest import *
from Tools.Serialize import Serialize
from Common.Analysis.Interpretability import Interpretability
from Common.Analysis.EvaluationMetrics import TypeML
from Models.BaseModel import BaseModel

DATASET_LEN = 100
DATASET_LEN_IDLIST = 4
METHODS = {
    'Lime': {
        'folder': "LIME",
        'n_files': [5],
        'deny_methods': []
    },
    'Lime': {
        'folder': "",
        'n_files': [4],
        'deny_methods': []
    },
    "Shapley": {
        'folder': "Shapley/png",
        'n_files': [5],
        'deny_methods': []
    },
    "Shapley": {
        'folder': "",
        'n_files': [3],
        'deny_methods': []
    },
    "IntegratedGradients": {
        'folder': "Integrated_Gradients/png",
        'n_files': [5],
        'deny_methods': []
    },
    "IntegratedGradients": {
        'folder': "",
        'n_files': [3],
        'deny_methods': []
    },
    'PermutationImportance': {
        'folder': "",
        'n_files': [4],
        'deny_methods': []
    },
    'PDP': {
        'folder': "PDP",
        'n_files': [DATASET_LEN_IDLIST],
        'deny_methods': []
    },
    'ALE': {
        'folder': "ALE",
        'n_files': [DATASET_LEN_IDLIST],
        'deny_methods': []
    },
    'Dice': {
        'folder': "DICE/png/",
        'n_files': [5],
        'deny_methods': ['ANN', 'RP', 'RLF']
    },
    'Dice': {
        'folder': "",
        'n_files': [3],
        'deny_methods': ['ANN', 'RP', 'RLF']
    },
    'RFPermutationImportance': {
        'folder': "",
        'n_files': [4],
        'deny_methods': []
    },
    'Anchor': {
        'folder': "Anchor",
        'n_files': [5],
        'deny_methods': []
    }
}

class TestInterpretability(BaseTest):

    def test_interpretability(self):
        lst_models = glob(FOLDER_MODELS_TEST + '*')

        self.assertTrue(len(lst_models) == NUMBER_MODELS, get_error_txt(ERROR_N_MODELS, FOLDER_MODELS_TEST))

        io_data = self.get_iodata()
        xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples = self.get_dataset(io_data)

        with open(join(FOLDER_TEST,'load_time.txt'),'w') as f:
            f.write('{}:{}:{}'.format('Load data',time.time(),time.time()*100))        

        for method, par in METHODS.items():
            for file_model in lst_models:
                model = BaseModel.load(file_model)
                name_model = basename(file_model).split("_")[0]

                if not name_model in par['deny_methods']:
                    params = {'type_ml': TypeML.CLASSIFICATION.value, 'model': name_model, 'classification_type': 'binary'}
                    p = FOLDER_TEST + name_model
                    args = Args(name_model)
                    cfg = self.get_config_holder(args, params, p)

                    BaseModel.save_model(cfg, model)
                    s = Serialize(model, xtr, ytr, xts, yts, id_list, cfg, io_data, idx_xts)
                    s.set_run_method(method)

                    Interpretability(s)
                    if par['folder'] != "":
                        lst_files = glob(join(FOLDER_TEST, par['folder']) + "/" + name_model + "*")
                    else:
                        lst_files = glob(join(FOLDER_TEST, name_model) + "_" + method + "*")
                    self.check_files(lst_files)
                    self.assertTrue(
                        len(lst_files) in par['n_files'],
                        get_error_txt(
                            ERROR_INTERPRETABILITY, "{} {}".format(
                                name_model, method +
                                " there are {} files when there should be {}".format(len(lst_files), par['n_files']))))

                    print('{}{:>24} {:<8}{}{:>3}{} '.format(bcolors.OKBLUE, method, name_model, bcolors.OKGREEN, "Ok",
                                                            bcolors.ENDC))


    def check_files(self, lst_files):
        for i in lst_files:
            self.assertNotEqual(getsize(i), 0, get_error_txt(ERROR_FILE_EMPTY, i))


if __name__ == "__main__":
    unittest.main()
