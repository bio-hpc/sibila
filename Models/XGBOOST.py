from .BaseModel import BaseModel
from os.path import join
import xgboost as xgb
from joblib import dump
from Tools.TypeML import TypeML
import matplotlib.pyplot as plt

PREFIX_OUT_XGBOOST = '{}_{}'  # Model, Dataset, numero de arboles, numero de profundidad, RANDOM_STATE

REMOVE_PARAMS_REGRESSOR = []


class XGBOOST(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(XGBOOST, self).__init__(io_data, cfg, id_list)

        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.model = xgb.XGBClassifier()
        elif self.cfg.get_params()['type_ml'].lower() == TypeML.REGRESSION.value:
            for i in REMOVE_PARAMS_REGRESSOR:
                if i in self.cfg.get_params()['params']:
                    del self.cfg.get_params()['params'][i]
                if i in self.cfg.get_params()['params_grid']:
                    del self.cfg.get_params()['params_grid'][i]
            self.model = xgb.XGBRegressor()

    def get_prefix(self):
        return join(self.cfg.get_folder(),
                    PREFIX_OUT_XGBOOST.format(
                        self.cfg.get_params()['model'],
                        self.cfg.get_name_dataset(),
                    ))

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.plotting()
        return ypr

    def plotting(self):
        file_ids = self.generate_file()
        _ = xgb.plot_tree(self.model, num_trees=0, fmap=file_ids)
        plt.savefig(self.get_prefix() + "_tree.png", dpi=300)
        plt.close()
        #_ = xgb.plot_importance(self.model, fmap=file_ids)
        #plt.savefig(self.get_prefix() + "_importance.png", dpi=300)

    def generate_file(self):
        """
            generates a file with id, value and data type that is passed to xgboost plotting methods.
            se i for indicator and q for quantity
            https://www.nuomiphp.com/eplan/en/34481.html
        """
        f = open(self.get_prefix() + "_feature_map.txt", 'w')
        for id in range(len(self.id_list)):
            f.write('{}\t{}\t{}\n'.format(id, self.id_list[id].replace(' ', '_'), 'q'))
        f.close()
        return self.get_prefix() + "_feature_map.txt"
