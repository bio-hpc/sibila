from .BaseModel import BaseModel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from os.path import join
from Tools.Graphics import Graphics
from Tools.ToolsModels import is_regression_by_args
from joblib import dump

PREFIX_OUT_DT = '{}_{}_{}'  # Model, Dataset, numero de arboles, numero de profundidad, RANDOM_STATE


class DT(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(DT, self).__init__(io_data, cfg, id_list)

        if not is_regression_by_args(self.cfg.get_args()):
            self.model = DecisionTreeClassifier(**self.cfg.get_params()['params'])

        elif is_regression_by_args(self.cfg.get_args()):
            if 'criterion' in self.cfg.get_params()['params']:
                del self.cfg.get_params()['params']['criterion']
            if 'criterion' in self.cfg.get_params()['params_grid']:
                del self.cfg.get_params()['params_grid']['criterion']

            #self.model = DecisionTreeRegressor(**self.cfg.get_params()['params'])
            self.model = DecisionTreeRegressor(max_depth=4, min_samples_split=5, max_leaf_nodes=10)

        else:
            print("Error: type_model not found ")
            exit()

    def get_prefix(self):
        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_DT.format(self.cfg.get_params()['model'], 
                                 self.cfg.get_name_dataset(),
                                 str(self.cfg.get_params()['n_jobs'])
                                )
        )

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        graphics = Graphics()
        graphics.graph_tree(self.model, self.id_list, self.targets, self.cfg.get_prefix())
        return ypr
