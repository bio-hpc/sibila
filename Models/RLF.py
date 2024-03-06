from .BaseModel import BaseModel
from os.path import join
from joblib import dump
from Tools.TypeML import TypeML
import numpy as np
from rulefit import RuleFit
from sklearn.ensemble import GradientBoostingRegressor

PREFIX_OUT_DT = '{}_{}'  # Model, Dataset


class RLF(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(RLF, self).__init__(io_data, cfg, id_list)
        self.model = RuleFit(**self.cfg.get_params()['params'])

    def get_prefix(self):
        return join(self.cfg.get_folder(),
                    PREFIX_OUT_DT.format(self.cfg.get_params()['model'], self.cfg.get_name_dataset()))

    def train(self, xtr, ytr):
        #self.model.fit(xtr, ytr, feature_names=self.id_list)
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        self.get_rules()
        ypr = np.round(ypr, 2)
        ypr = np.array(ypr).astype(int)
        return ypr

    def get_rules(self):
        rules = self.model.get_rules()
        rules = rules[rules.coef != 0].sort_values("support", ascending=False)
        rules.to_csv(self.cfg.get_prefix() + "_rules.csv")

