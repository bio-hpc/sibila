from Models.Utils.TrainGrid import TrainGrid
from Models.Utils.LearningHistoryCallback import LearningHistoryCallback
from Models.Utils.CrossValidation import CrossValidation
import abc
from Tools.DatasetBalanced import DatasetBalanced
from Tools.ToolsModels import is_tf_model, is_regression_by_config, is_xgboost_model, is_ripper_model, is_rulefit_model
import tensorflow as tf
import os
from joblib import dump
from os.path import splitext
from joblib import load
from Tools.IOData import IOData
import pickle
import numpy as np

class BaseModel(abc.ABC):
    RANDOM_STATE = 500
    N_ITER = 4  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    N_JOBS = 4  # Number of jobs to run in parallel. None means 1 unless in

    def __init__(self, io_data, cfg, id_list):
        self.io_data = io_data
        self.cfg = cfg
        self.id_list = id_list
        self.targets = []

    def __has_class_weights(self):
        declined_models = ['KNeighbors', 'Bagging', 'RuleFit']
        is_declined_model = False

        for s in declined_models:
            if s in str(self.model):
                is_declined_model = True

        if not self.cfg.get_args()['regression'] and not is_declined_model:
            return True
        return False

    def __has_seed(self):
        declined_models = ['SVR', 'KNeighbors', 'LinearRegression']
        is_declined_model = False

        for s in declined_models:
            if s in str(self.model):
                is_declined_model = True

        if not is_declined_model:
            return True
        return False

    def model_fit(self, xtr, ytr):
        self.io_data.print_m('\n\tStart Train {}'.format(self.cfg.get_params()['model']))

        self.targets = np.unique(ytr).astype(str)

        class_weights = DatasetBalanced.get_class_weights(self.model, ytr, self.cfg)
        if is_tf_model(self.model) and is_regression_by_config(self.cfg):
            class_weights = None

        if 'train_grid' in self.cfg.get_params().keys() and self.cfg.get_params()['train_grid'].upper() != "NONE":
            cvmethod = CrossValidation(self.io_data).choice_method('GKF')
            cvarg = self.cfg.get_args()['crossvalidation']
            if cvarg is not None:
                cv = CrossValidation(self.io_data)
                cvmethod = cv.choice_method(self.cfg.get_args()['crossvalidation'])

            train_grid = TrainGrid(cvmethod)

            func = getattr(train_grid, self.cfg.get_params()['train_grid'])
            try:
                self.cfg.get_params()['params'] = func(self.model, self.cfg.get_params()['params_grid'], xtr, ytr)
                self.model.set_params(**self.cfg.get_params()['params'])
            except Exception as e:
                self.io_data.print_m("ERROR No hyperparameters search will be performed for {}".format(self.cfg.get_params()['model']))
                pass

        if is_tf_model(self.model):
            self.model.fit(xtr,
                           ytr,
                           verbose = 1,
                           batch_size = self.cfg.get_params()['params']['batch_size'],
                           epochs = self.cfg.get_params()['params_grid']['epochs'],
                           class_weight = class_weights,
                           callbacks = [ 
                               tf.keras.callbacks.TerminateOnNaN(),
                               tf.keras.callbacks.ReduceLROnPlateau(),
                               LearningHistoryCallback(self.cfg) 
                           ]
            )
        else:
            params_model = self.model.get_params()
            if self.__has_seed():
                # svr and knn are the only models that do not support random_state
                params_model['random_state'] = self.cfg.get_args()['seed']
            if self.__has_class_weights():
                params_model['class_weight'] = class_weights
            if is_ripper_model(self.model):
                params_model['feature_names'] = self.id_list

            self.model.set_params(**params_model)
            self.model.fit(xtr, ytr)

        self.io_data.print_m('End Train {}'.format(self.cfg.get_params()['model']))

    def model_predict(self, xts):
        self.io_data.print_m('\n\tStart Predict {}'.format(self.cfg.get_params()['model']))
        ypr = self.model.predict(xts)
        self.io_data.print_m('End Predict {}'.format(self.cfg.get_params()['model']))
        self.cfg.set_time_end()
        try:
            self.cfg.set_model_params(self.model.get_params(True))
        except:
            self.io_data.print_m("ERROR {} does not have the method, not all parameters can be displayed".format(
                self.cfg.get_params()['model']))
        return ypr
    
    def predict_proba(self, X):
        """
        Predice probabilidades incluso para modelos que no tienen soporte nativo para predict_proba.
        """
        if hasattr(self.model, "predict_proba"):
            # Si el modelo ya tiene predict_proba, úsalo directamente
            return self.model.predict_proba(X)
        else:
            # Estimar probabilidades manualmente
            pred = self.model.predict(X)
            # Convertir predicciones binarias a formato probabilístico
            proba = np.vstack([1 - pred, pred]).T
            return proba


    def get_model(self):
        return self.model

    @abc.abstractmethod
    def train(self, xtr, ytr):
        """"""

    @abc.abstractmethod
    def get_prefix(self):
        """mandatory method to return the prefix"""

    @abc.abstractmethod
    def predict(self, xts, idx_xts):
        """"""

    @staticmethod
    def load(filename):
        """load the model from a file"""
        name, extension = os.path.splitext(filename)

        if extension == '.h5':
            return tf.keras.models.load_model(filename)
        elif extension == '.joblib':
            return load(filename)
        elif extension == '.dat':
            return pickle.load(open(filename, "rb"))
        else:
            IOData.print_e('File format {} not recognized'.format(extension))
            exit(-1)

    @staticmethod
    def get_filename_save_model(cfg, model):

        if is_tf_model(model):
            return cfg.get_prefix() + '.h5'
        elif is_xgboost_model(model):
            return cfg.get_prefix() + '.dat'
        else:
            return cfg.get_prefix() + '.joblib'

    @staticmethod
    def save_model(cfg, model):
        file_out = BaseModel.get_filename_save_model(cfg, model)

        if is_tf_model(model):
            model.save(file_out)
        elif is_xgboost_model(model):
            pickle.dump(model, open(file_out, "wb"))
        else:
            dump(model, file_out)
