from .BaseModel import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from os.path import join
from Tools.Graphics import Graphics
from joblib import dump
from Tools.ToolsModels import is_regression_by_args

PREFIX_OUT_LR = '{}_{}_{}'  # Model, Dataset, numero de arboles, numero de profundidad, RANDOM_STATE


class LR(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(LR, self).__init__(io_data, cfg, id_list)

        if not is_regression_by_args(self.cfg.get_args()):
            self.model = LogisticRegression(**self.cfg.get_params()['params'])
        elif is_regression_by_args(self.cfg.get_args()):
            self.model = LinearRegression()
        else:
            print("Error: type_model not found ")
            exit()

    def get_prefix(self):
        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_LR.format(self.cfg.get_params()['model'], self.cfg.get_name_dataset(),
                                 str(self.cfg.get_params()['params']['n_jobs'])))

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        return ypr
