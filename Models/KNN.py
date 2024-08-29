from os.path import join
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from joblib import dump
from .BaseModel import *
from Tools.Graphics import Graphics
from Tools.TypeML import TypeML
from sklearn.inspection import permutation_importance

PREFIX_OUT_KNN = '{}_{}_{}_{}'


class KNN(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(KNN, self).__init__(io_data, cfg, id_list)

        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.model = KNeighborsClassifier(**self.cfg.get_params()['params'])
        elif self.cfg.get_params()['type_ml'].lower() == TypeML.REGRESSION.value:
            self.model = KNeighborsRegressor(**self.cfg.get_params()['params'])
        else:
            print("Error: type_model not found ")
            exit()

    def get_prefix(self):
        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_KNN.format(self.cfg.get_params()['model'], self.cfg.get_name_dataset(),
                                  self.cfg.get_params()['params']['n_jobs'],
                                  str(self.cfg.get_params()['params']['weights'])
                                 )
        )

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)
        graphics = Graphics()
        file_out = self.cfg.get_prefix() + '_points.png'
        graphics.graph_knn_points(self.model, xtr, ytr, self.id_list, file_out)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        return ypr
