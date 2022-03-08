#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

from sklearn.base import BaseEstimator
import numpy as np


class RipperEstimator(BaseEstimator):
    def __init__(self, rp_model, Y=None, cfg=None):
        self.rp_model = rp_model
        self._estimator_type = 'classifier'
        self.cfg = cfg
        self.classes_ = []
        self.Y = Y
        if Y is not None:
            self.classes_ = np.unique(Y)

    def fit(self, X, Y):
        self.rp_model.fit(X, Y)

    def predict_proba(self, X):
        return np.array(self.rp_model.predict(X.values)).astype(int)

    def decision_function(self, X):
        return self.rp_model.predict_proba(X.values)
