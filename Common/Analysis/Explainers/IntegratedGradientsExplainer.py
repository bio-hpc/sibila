#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
import numpy as np
import shap
from Tools.ToolsModels import is_tf_model, is_ripper_model, is_regression_by_config
from Tools.Graphics import Graphics
from alibi.explainers import IntegratedGradients
from Tools.Estimators.SklearnNetwork import SklearnNetwork
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Common.Config.ConfigHolder import FEATURE, ATTR, STD, PROBA, TRUEVAL, PREDVAL

class IntegratedGradientsExplainer(ExplainerModel):

    def explain(self):
        # Get numerical feature importances with the integrated gradients technique
        # https://docs.seldon.io/projects/alibi/en/latest/methods/IntegratedGradients.html#Examples
        # https://distill.pub/2020/attribution-baselines/
        if is_tf_model(self.model):
            self.xts_ = self.xts
            model_ = self.model
            target_ = self.model(self.xts).numpy().argmax(axis=1).reshape(-1, 1)
        elif is_ripper_model(self.model):
            self.xts_ = self.xts
            model_ = SklearnNetwork(self.model, self.xts_.shape)
            target_ = np.array(self.model.predict(self.xts)).astype(int)
        else:
            self.xts_ = self.xts
            model_ = SklearnNetwork(self.model, self.xts.shape)
            target_ = self.model.predict(self.xts)

        baseline_ = self.get_baseline(self.xts_)

        ig = IntegratedGradients(model_, method='riemann_trapezoid', n_steps=50)
        explanation = ig.explain(self.xts_, baselines=baseline_, target=target_)
        self.attrs = np.squeeze(explanation.attributions)

        # global explanation
        return pd.DataFrame({FEATURE: self.id_list, ATTR: np.mean(self.attrs, axis=0), STD: np.std(self.attrs, axis=0)})

    def get_baseline(self, X):
        return shap.sample(X, X.shape[0])*1.005

    def plot(self, df, method=None):
        title = 'Integrated Gradients'

        # global explanation
        Graphics().plot_attributions(df, title, self.prefix + "_" + method + ".png", errors=self.get_errors(df))

        # local explanations
        for i in tqdm(range(self.attrs.shape[0])):
            filename = "{}_IntegratedGradients_{}".format(Path(self.cfg.get_prefix()).stem, self.idx_xts[i])
            path_csv = "{}csv/{}.csv".format(self.io_data.get_integrated_gradients_folder(), filename)
            path_png = "{}png/{}.png".format(self.io_data.get_integrated_gradients_folder(), filename)
            proba = self.proba_sample(self.xts[i])
            
            # Sort in ascending order for plotting correctly
            if is_regression_by_config(self.cfg):
                df2 = pd.DataFrame({FEATURE: self.id_list, ATTR: self.attrs[i], TRUEVAL: self.yts[i], PREDVAL: proba})
            else:
                df2 = pd.DataFrame({FEATURE: self.id_list, ATTR: self.attrs[i], PROBA: proba})
            df2 = df2.reindex(df2[ATTR].abs().sort_values(ascending=False).index)

            self.io_data.save_dataframe_cols(df2, df2.columns, path_csv)

            # Add the real value into the label            
            df2[FEATURE] = df2[FEATURE].apply(lambda x: '{:.3f}={}'.format(self.get_value(x, i), x))

            # Take the N most important features and sum up all the rest
            df2 = self.summarize(df2)

            Graphics().plot_attributions(df2, title, path_png, self.idx_xts[i])
            del df2

    def get_value(self, feature, row_id):
        index = self.id_list.index(feature)
        return self.xts_[row_id, index]

