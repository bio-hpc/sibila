#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class XGBOOSTRegressor(GradientBoostingRegressor):
    def __init__(self, xgb_model, Y=None, cfg=None):
        super().__init__()
        self.xgb_model = xgb_model
        self._estimator_type = 'regressor'
        self.cfg = cfg
        self.Y = Y

    def fit(self, X, y, sample_weight=None, monitor=None):
        super().fit(X, y) # do this to set the internal state
        self.xgb_model.fit(X, y)
        return self

    def predict(self, X):
        return np.array(self.xgb_model.predict(X.values))

    def predict_proba(self, X):
        return np.array(self.xgb_model.predict(X.values))
