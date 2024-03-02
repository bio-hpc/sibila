# Import svm model
from sklearn import svm
from .BaseModel import *
from os.path import join
from joblib import dump
from Tools.TypeML import TypeML

#### kernel "linear", "poly", "rbf", "sigmoid", "precomputed", default="rbf"
PREFIX_OUT_SVM = '{}_{}_{}'  # Model, Dataset, numero de arboles, numero de profundidad, RANDOM_STATE
REMOVE_PARAMS_REGRESSOR = ['decision_function_shape', 'probability']


class SVM(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(SVM, self).__init__(io_data, cfg, id_list)

        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.model = svm.SVC(**self.cfg.get_params()['params'])
        elif self.cfg.get_params()['type_ml'].lower() == TypeML.REGRESSION.value:
            for i in REMOVE_PARAMS_REGRESSOR:
                if i in self.cfg.get_params()['params']:
                    del self.cfg.get_params()['params'][i]
                if i in self.cfg.get_params()['params_grid']:
                    del self.cfg.get_params()['params_grid'][i]
            self.model = svm.SVR(**self.cfg.get_params()['params'])
        else:
            print("Error: type_model not found ")
            exit()

    def get_prefix(self):
        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_SVM.format(self.cfg.get_params()['model'], 
                                  self.cfg.get_name_dataset(),
                                  self.cfg.get_params()['n_jobs']
                                 )
        )

    def train(self, xtr, ytr):
        #self.model.fit(xtr, ytr)
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        return ypr
