import shutil
import os
import unittest
from Common.Config.ConfigHolder import ConfigHolder
from glob import glob
from Tests.errors import get_error, get_error_txt
from Tools.datasets import get_dataset, split_samples
from Tools.IOData import IOData
from os.path import isdir, join, isfile

#
#   Ignore warning TF
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FILE_DATASET = 'Datasets/Tests/clasificacion-sintetico_v1.csv'
FOLDER_MODELS_TEST = 'Tests/Models/'
FOLDER_TEST = 'Tests/WorkingDir/'
FOLDER_TEST_NUMBER = 9
NUMBER_MODELS = 10
SEED = 2021
SPLIT_DATASET = 0.95

ERROR_FILE = 'E0001'
ERROR_N_MODELS = 'E0003'
ERROR_N_CREATE_FOLDER = 'E0004'
ERROR_READ_DATASET = 'E0005'
ERROR_INTERPRETABILITY = 'E0006'
ERROR_FILE_EMPTY = 'E0007'
ERROR_SAVED_MODEL = 'E0100'
ERROR_CONF_MATRIX = 'E0102'
ERROR_ROC_PROBA_CLASS = 'E0103'
ERROR_DATA_JSON = 'E0104'
ERROR_DATA_TXT = 'E0105'
ERROR_ROC_PROBA = 'E0106'
ERROR_ROC = 'E0107'
ERROR_DATA_CSV = 'E0108'
ERROR_MODEL = 'E0109'
ERROR_NORMALIZATION = 'E0110'
ERROR_BALANCING = 'E0111'
ERROR_NOT_BINARY = 'E0112'
ERROR_DIFF_LENGTH = 'E0113'
ERROR_CORRELATION = 'E0114'


class Args:
    def __init__(self, model):
        self.queue = True
        self.seed = SEED
        self.dataset = FILE_DATASET
        self.folder = FOLDER_TEST
        self.model = model
        self.regression = False
        self.balanced = None
        self.crossvalidation = None
        self.parameters = None
        self.normalize = None
        self.balanced = None

class BaseTest(unittest.TestCase):

    def setUp(self):
        # cleanup working directory
        if isdir(FOLDER_TEST):
            shutil.rmtree(FOLDER_TEST)
        os.mkdir(FOLDER_TEST)

    def tearDown(self):
        if isdir(FOLDER_TEST):
            shutil.rmtree(FOLDER_TEST)
            
    def count_files_by_pattern(self, pattern):
        files = [ foo for foo in glob(join(FOLDER_TEST, pattern)) ]
        return len(files)

    def get_iodata(self):
        io_data = IOData()
        io_data.create_dirs(FOLDER_TEST)
        aux = glob(FOLDER_TEST + "*")
        aux = [ x for x in aux if "out" not in x ] # exclude "out" folder
        self.assertTrue(len(aux) == FOLDER_TEST_NUMBER, get_error_txt(ERROR_N_CREATE_FOLDER, aux))
        return io_data

    def get_dataset(self, io_data):
        self.assertTrue(isfile(FILE_DATASET), get_error_txt(ERROR_FILE, FILE_DATASET))
        x, y, id_list, idx_samples, target_classes = get_dataset(FILE_DATASET, io_data)

        DATASET_LEN = x.shape[0]
        DATASET_LEN_IDLIST = x.shape[1]

        self.assertTrue(len(x) == DATASET_LEN, get_error_txt(ERROR_READ_DATASET, "X"))
        self.assertTrue(len(y) == DATASET_LEN, get_error_txt(ERROR_READ_DATASET, "Y"))
        self.assertTrue(len(id_list) == DATASET_LEN_IDLIST, get_error_txt(ERROR_READ_DATASET, "id_list"))
        self.assertTrue(len(idx_samples) == DATASET_LEN, get_error_txt(ERROR_READ_DATASET, "idx_samples"))

        xtr, xts, ytr, yts, idx_xtr, idx_xts = split_samples(x, y, SPLIT_DATASET, io_data, SEED, idx_samples)

        self.assertTrue(len(xtr) == DATASET_LEN * SPLIT_DATASET, get_error_txt(ERROR_READ_DATASET, "xtr"))
        self.assertTrue(len(ytr) == DATASET_LEN * SPLIT_DATASET, get_error_txt(ERROR_READ_DATASET, "ytr"))
        self.assertTrue(len(idx_xtr) == DATASET_LEN * SPLIT_DATASET, get_error_txt(ERROR_READ_DATASET, "idx_xtr"))

        cutoff = round((1 - SPLIT_DATASET), 2)
        self.assertTrue(len(xts) == DATASET_LEN * cutoff, get_error_txt(ERROR_READ_DATASET, "xts"))
        self.assertTrue(len(yts) == DATASET_LEN * cutoff, get_error_txt(ERROR_READ_DATASET, "yts"))
        self.assertTrue(len(idx_xts) == DATASET_LEN * cutoff, get_error_txt(ERROR_READ_DATASET, "idx_xts"))
        return xtr, xts, ytr, yts, idx_xtr, idx_xts, id_list, idx_samples

    def get_config_holder(self, args, params, prefix):
        cfg = ConfigHolder(FILE_DATASET, FOLDER_TEST, args, params)
        cfg.set_prefix(prefix)
        cfg.set_cores(1)
        return cfg
