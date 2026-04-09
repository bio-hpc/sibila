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
import tensorflow as tf
from Tools.ToolsModels import is_regression_by_args, is_multiclass
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

        if is_regression_by_args(self.cfg.get_args()):
            task = "regression"
            n_classes = None
        else:
            task = "classification"
            n_classes = self.cfg.get_n_classes()

        self.xts_ = np.array(self.xts).astype(np.float32)
        if len(self.xts_.shape) == 1:
            self.xts_ = self.xts_.reshape(1, -1)

        baseline_ = self.get_baseline(self.xts_)
        if baseline_.shape != self.xts_.shape:
            baseline_ = np.zeros_like(self.xts_)

        model_ = SklearnNetwork(self.model, input_dim=self.xtr.shape[1], task=task, n_classes=n_classes)
        model_.prepare(self.xtr)

        inputs_tf = tf.convert_to_tensor(self.xts, dtype=tf.float32)
        preds = model_(inputs_tf).numpy()

        #y_true = model_.original_model.predict(self.xts)
        #y_surr = model_.surrogate.predict(self.xts)
        #print("Ejemplo real vs surrogate:")
        #print(np.c_[y_true[:10], y_surr[:10]])

        if task == "classification":
            if preds.shape[1] == 1:
                target_ = (preds > 0.5).astype(int).reshape(-1)
            else:
                target_ = np.argmax(preds, axis=1)
        else:
            target_ = None

        ig = IntegratedGradients(model_, method='riemann_trapezoid', n_steps=50)
        explanation = ig.explain(self.xts_, baselines=baseline_, target=target_)
        self.attrs = np.squeeze(explanation.attributions)

        # global explanation
        return pd.DataFrame({FEATURE: self.id_list, ATTR: np.mean(self.attrs, axis=0), STD: np.std(self.attrs, axis=0)})

    def get_baseline(self, X):
        return np.mean(self.xtr, axis=0, keepdims=True)

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
            if is_regression_by_args(self.cfg.get_args()):
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
