#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import numpy as np
from sklearn.base import BaseEstimator
from Tools.ToolsModels import is_regression_by_config

class TensorFlowEstimator(BaseEstimator):

    def __init__(self, inner_model, cfg=None, Y=None):
        super().__init__()
        self.inner_model_ = inner_model
        self.fitted_ = False
        self._estimator_type = 'regressor' if is_regression_by_config(cfg) else 'classifier'
        self.cfg_ = cfg
        self.classes_ = None
        if self._estimator_type == 'classifier' and not Y is None:
            self.classes_ = np.unique(Y)
    
    def fit(self, X, Y):
        self.X_ = X
        self.y_ = Y

        if not self.fitted_:
            if self.cfg_ is not None and 'params' in self.cfg_.get_params().keys():
                self.inner_model_.fit(X, Y, epochs=self.cfg_.get_params()['params']['epochs'])
            else:
                self.inner_model_.fit(X, Y)

            self.fitted_ = True

        return self

    def predict(self, X):
        return self.__predict(X)

    def predict_proba(self, X):
        return self.__predict(X)

    def __predict(self, X):
        return self.inner_model_.predict(X)

    def get_params(self, deep=True):
        return {'inner_model_': self.inner_model_, 'fitted_': self.fitted_, 'cfg_': self.cfg_}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __str__(self):
        return 'TensorFlowEstimator(_estimator_type: {}, cfg_: {}, classes_: {}, fitted_: {}, inner_model_: {})'.format(self._estimator_type, self.cfg_, self.classes_, self.fitted_, self.inner_model_)
