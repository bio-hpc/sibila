#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np


class RuleFitEstimator(BaseEstimator):
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

        #return self.rp_model.predict_proba(X)
        #if "values" in X:
        try:
            ypr = self.rp_model.predict(X.values)
        except:
            ypr = self.rp_model.predict(X)
            ypr = np.round(ypr, 3)
            ypr = np.array(ypr)

        return ypr

    def decision_function(self, X):
        return self.predict_proba(X)  #rp_model.predict_proba(X.values)
