from os.path import join
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump
from .BaseModel import *
from Tools.Graphics import Graphics
from Tools.TypeML import TypeML

PREFIX_OUT_RF = '{}_{}_{}_{}'
REMOVE_PARAMS_REGRESSOR = ['criterion']


class RF(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(RF, self).__init__(io_data, cfg, id_list)
        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.model = RandomForestClassifier(**self.cfg.get_params()['params'])
        elif self.cfg.get_params()['type_ml'].lower() == TypeML.REGRESSION.value:
            for i in REMOVE_PARAMS_REGRESSOR:
                if i in self.cfg.get_params()['params']:
                    del self.cfg.get_params()['params'][i]
                if i in self.cfg.get_params()['params_grid']:
                    del self.cfg.get_params()['params_grid'][i]

            self.model = RandomForestRegressor(**self.cfg.get_params()['params'])
        else:
            print("Error: type_model not found")
            exit()

    def get_prefix(self):

        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_RF.format(self.cfg.get_params()['model'], 
                                 self.cfg.get_name_dataset(),
                                 self.cfg.get_params()['params']['n_jobs'],
                                 self.cfg.get_params()['params']['verbose']
                                )
        )

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        #self.graph_tree(self.get_prefix(), self.model.estimators_[0], self.id_list)
        return ypr

    def graph_tree(self, prefix, model, id_list):
        g = Graphics()
        g.graph_tree(model, id_list, self.targets, prefix)
