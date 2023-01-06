#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Jorge de la Peña"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jpena@ucam.edu"
__status__ = "Production"

import abc
import numpy as np
import pandas as pd
from Tools.Estimators.TensorFlowEstimator import TensorFlowEstimator
from Tools.ToolsModels import is_tf_model
from Common.Config.ConfigHolder import ATTR, FEATURE, MAX_IMPORTANCES, STD

class ExplainerModel(abc.ABC):

    def __init__(self, model, xtr, ytr, xts, yts, id_list, cfg, io_data, idx_xts):
        self.io_data = io_data
        self.model = model
        self.xtr = xtr
        self.ytr = ytr
        self.xts = xts
        self.yts = yts
        self.idx_xts = idx_xts
        self.cfg = cfg
        self.prefix = cfg.get_prefix()
        self.id_list = id_list
        self.random_state = cfg.get_args()['seed']
        self.class_target = np.unique(ytr).astype(str)
        self.estimator = TensorFlowEstimator(inner_model=model, cfg=cfg, Y=ytr)

    @abc.abstractmethod
    def explain(self):
        """
        """

    @abc.abstractmethod
    def plot(self, df, method=None):
        """
        """

    def summarize(self, df):
        if len(df) > MAX_IMPORTANCES:
            others_sum = df[MAX_IMPORTANCES:][ATTR].sum()
            n_others = len(self.id_list) - MAX_IMPORTANCES
            title = 'Sum other {} features'.format(str(n_others))
            df_others = pd.DataFrame(data=[[title, others_sum]], columns=[FEATURE, ATTR])

            df = df[:MAX_IMPORTANCES]
            df = pd.concat([df[:MAX_IMPORTANCES], df_others], ignore_index=True)
        return df

    def get_errors(self, df):
        errors = []
        for c in df[FEATURE].to_numpy():
            _ = df.loc[df[FEATURE] == c][STD].to_numpy()
            errors.append(_[0] if len(_) > 0 else 0.0)
        return errors

    def proba_sample(self, x):
        return np.amax(self.model.predict_proba(np.array([x])))

