#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from Tools.ToolsModels import is_regression_by_config
from Tools.Graphics import Graphics
import numpy as np
import re
from Tools.HTML.LIMEHTMLBuilder import LIMEHTMLBuilder
from lime import lime_tabular
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Tools.ToolsModels import is_rulefit_model
from Common.Config.ConfigHolder import ATTR, COLNAMES, FEATURE, STD

class LimeExplainer(ExplainerModel):

    def explain(self):
        """
         https://github.com/marcotcr/lime
         https://lime-ml.readthedocs.io/en/latest/index.html
        """
        return self.execute()

    def execute(self):
        if not is_regression_by_config(self.cfg):
            explainer, predict_fn, n_samples = self.lime_classification()
        else:
            explainer, predict_fn, n_samples = self.lime_regression()

        # local interpretation
        prefix = Path(self.cfg.get_prefix()).stem
        self.html = LIMEHTMLBuilder()
        df_local = []

        for i in tqdm(range(len(self.xts))):
            x = self.xts[i]
            exp = explainer.explain_instance(x, predict_fn, num_features=len(self.id_list), top_labels=1, num_samples=n_samples)
            if is_regression_by_config(self.cfg):
                ypr = exp.predicted_value
                explanation = exp.as_list(0)
            else:
                ypr = exp.available_labels()[0]
                explanation = exp.as_list(ypr)

            df_aux = self.lime_save_explanations(explanation, prefix, self.idx_xts[i], ypr)
            df_local.append(df_aux)
            self.html.append(exp.as_html(), sample_id=self.idx_xts[i])
            del df_aux

        self.html.close()

        # averaged attributions
        self.df_full = pd.concat(df_local, ignore_index=True)
        self.df_std = self.df_full.groupby(FEATURE).std().reset_index()
        df_global = self.df_full.groupby(FEATURE).mean().reset_index()

        return df_global

    def plot(self, df, method=None):
        # local explanations
        Graphics().plot_lime_html(self.html.get(), self.cfg.get_prefix() + '_Lime_tabular_explainer.html')

        # global explanation
        errors = []
        for c in df[FEATURE].to_numpy():
            _ = self.df_std.loc[self.df_std[FEATURE] == c][ATTR].to_numpy()
            errors.append(_[0] if len(_) > 0 else 0.0)

        result = self.df_full.drop(columns=['range','class']).groupby([FEATURE], as_index=False).agg(['mean', 'std'])
        result = result.droplevel(level=0, axis=1).reset_index()
        result.columns = COLNAMES

        self.io_data.save_dataframe_cols(result, result.columns, self.cfg.get_prefix() + '_Lime.csv')
        Graphics().plot_attributions(df, 'LIME', self.cfg.get_prefix() + '_Lime.png', errors=errors)

    def lime_classification(self):
        explainer = lime_tabular.LimeTabularExplainer(self.xtr,
                                                      feature_names=self.id_list,
                                                      class_names=np.unique(self.yts, axis=0).astype(str),
                                                      discretize_continuous=True,
                                                      discretizer='entropy',
                                                      training_labels=self.ytr,
                                                      random_state=self.random_state,
                                                      feature_selection='forward_selection')
        return explainer, self.model.predict_proba, 5000

    def lime_regression(self):
        def lime_predict(x):
            pred = self.model.predict(x)
            return pred.reshape(pred.shape[0])

        explainer = lime_tabular.LimeTabularExplainer(self.xtr,
                                                      feature_names=self.id_list,
                                                      discretize_continuous=True,
                                                      mode='regression')
        return explainer, lime_predict, 5000

    def lime_save_explanations(self, explanation, prefix, index, ypr):
        def get_feature_name(e):
            m = re.split('[<]+ | [>]+ | [<=]+ | [>=]+ | [=]+', e)
            return m[1] if len(m) > 2 else m[0]

        data = [[get_feature_name(e[0]), e[1], e[0], ypr] for e in explanation]
        df = pd.DataFrame(data=data, columns=[FEATURE, ATTR] + ['range', 'class'])
        self.io_data.save_dataframe_cols(
            df, df.columns,
            self.io_data.get_lime_folder() + "{}_Lime_explain_{}.csv".format(prefix, index))

        return df
