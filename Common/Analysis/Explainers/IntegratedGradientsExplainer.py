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
from Tools.ToolsModels import is_tf_model, is_ripper_model
from Tools.Graphics import Graphics
from alibi.explainers import IntegratedGradients
from Tools.Estimators.SklearnNetwork import SklearnNetwork
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel


class IntegratedGradientsExplainer(ExplainerModel):

    CSV_COLUMNS = ['feature', 'weight']

    def explain(self):
        # Get numerical feature importances with the integrated gradients technique
        # https://docs.seldon.io/projects/alibi/en/latest/methods/IntegratedGradients.html#Examples
        # https://distill.pub/2020/attribution-baselines/
        #if self.cfg.get_params()['model'] == 'RNN':
        #    model_ = model
        #    target_ = model(xts).numpy().argmax(axis=1)
        #    xts_ = tf.squeeze(xts).numpy()
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
        df = pd.DataFrame(self.attrs, columns=self.id_list)

        df_mean = df.mean().to_frame().reset_index()
        self.df_std = df.std().to_frame().reset_index()
        self.df_std.columns = self.CSV_COLUMNS

        df_mean.columns = self.CSV_COLUMNS
        return df_mean

    def get_baseline(self, X):
        #return shap.sample(X, X.shape[0])*1.101
        return shap.sample(X, X.shape[0])*1.005

    def plot(self, df, method=None):
        title = 'Integrated Gradients'

        errors = []
        for c in df['feature'].to_numpy():
            _ = self.df_std.loc[self.df_std['feature'] == c]['weight'].to_numpy()
            errors.append(_[0] if len(_) > 0 else 0.0)

        self.io_data.save_dataframe_cols(df, df.columns, self.prefix + '_IntegratedGradients.csv')
        Graphics().plot_attributions(df, title, self.prefix + "_IntegratedGradients.png", errors=errors)

        # local explanations
        for i in tqdm(range(self.attrs.shape[0])):
            filename = "{}_IntegratedGradients_{}".format(Path(self.cfg.get_prefix()).stem, self.idx_xts[i])
            path_csv = "{}csv/{}.csv".format(self.io_data.get_integrated_gradients_folder(), filename)
            path_png = "{}png/{}.png".format(self.io_data.get_integrated_gradients_folder(), filename)
            
            # Sort in ascending order for plotting correctly
            df2 = pd.DataFrame({'feature': self.id_list, 'weight': self.attrs[i]})
            df2 = df2.reindex(df2['weight'].abs().sort_values(ascending=False).index)

            self.io_data.save_dataframe_cols(df2, df2.columns, path_csv)

            # Add the real value into the label            
            df2['feature'] = df2['feature'].apply(lambda x: '{:.3f}={}'.format(self.get_value(x, i), x))

            # Take the N most important features and sum up all the rest
            df2 = self.summarize(df2)

            Graphics().plot_attributions(df2, title, path_png, self.idx_xts[i])
            del df2

    def get_value(self, feature, row_id):
        index = self.id_list.index(feature)
        return self.xts_[row_id, index]
