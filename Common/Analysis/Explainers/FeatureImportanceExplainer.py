#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import eli5
import pandas as pd
from Tools.Graphics import Graphics
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel


class FeatureImportanceExplainer(ExplainerModel):
    def explain(self):
        """
            https://eli5.readthedocs.io/en/latest/index.html
        """

        explain_weights = eli5.explain_weights(self.model, feature_names=self.id_list, top=999)

        df_explain_weights = eli5.formatters.format_as_dataframe(explain_weights)
        df = pd.DataFrame({
            'feature': df_explain_weights['feature'],
            'weight': df_explain_weights['weight'],
        }) if df_explain_weights is not None else None
        return df

    def plot(self, df, method=None):
        Graphics().graphic_pie(df, self.prefix + '_' + method + '_hist.png', 'Feature Importance eli5 (Pie)')
        Graphics().graph_hist(df, self.prefix + '_' + method + '_pie.png', 'Feature Importance eli5 (Bar)')

