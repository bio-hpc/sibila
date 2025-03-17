#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from Tools.ToolsModels import is_tf_model, is_ripper_model, is_regression_by_config
import shap
from Tools.Graphics import Graphics
import numpy as np
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Common.Config.ConfigHolder import FEATURE, ATTR, PROBA, TRUEVAL, PREDVAL

class ShapleyExplainer(ExplainerModel):

    def explain(self):
        """
         https://shap-lrjball.readthedocs.io/en/latest/generated/shap.Explainer.html
         https://github.com/slundberg/shap/blob/master/shap/explainers/
        """
        def shapley_predict(x):
            pred = self.model.predict(x)
            return pred.argmax(axis=1)

        def ripper_predict(x):
            pred = self.model.predict(x)
            return np.array(pred).astype(int)

        # global explanation
        #K = len(np.unique(self.ytr, axis=0))
        #x_summary = shap.kmeans(self.xtr, K)
        model_fn = shapley_predict if is_tf_model(self.model) else self.model.predict
        model_fn = ripper_predict if is_ripper_model(self.model) else model_fn

        background = shap.maskers.Independent(self.xtr)
        explainer = shap.Explainer(model_fn, background)
        self.shap_values = explainer(self.xts)

        added_values = np.absolute(self.shap_values.values).sum(axis=0)

        overall_values = dict(zip(self.id_list, added_values))
        self.feature_names = ['{} [{}]'.format(f, round(overall_values[f], 3)) for f in self.id_list]
        return pd.DataFrame({FEATURE:self.id_list, ATTR:added_values})

    def plot(self, df, method=None):
        # global explanation
        Graphics().plot_shapley(self.xts, self.feature_names, self.shap_values, self.prefix)

        # local explanations
        prefix = Path(self.cfg.get_prefix()).stem
        for i in tqdm(range(len(self.xts))):
            proba = self.proba_sample(self.xts[i])

            if is_regression_by_config(self.cfg):
                df_aux = pd.DataFrame({FEATURE: self.id_list, ATTR: self.shap_values[i].values, 'value': self.xts[i], TRUEVAL: self.yts[i], PREDVAL: proba})
            else:
                df_aux = pd.DataFrame({FEATURE: self.id_list, ATTR: self.shap_values[i].values, 'value': self.xts[i], PROBA: proba})

            self.io_data.save_dataframe_cols(
                df_aux, df_aux.columns,
                self.io_data.get_shapley_folder() + 'csv/{}_Shapley_{}.csv'.format(prefix, self.idx_xts[i]))

            Graphics().plot_shapley_local(
                self.shap_values[i], self.id_list,
                self.io_data.get_shapley_folder() + 'png/{}_Shapley_{}.png'.format(prefix, self.idx_xts[i]),
                self.idx_xts[i])
