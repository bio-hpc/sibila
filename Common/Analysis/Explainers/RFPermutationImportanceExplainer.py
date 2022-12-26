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
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Common.Config.ConfigHolder import FEATURE, ATTR, STD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RFPermutationImportanceExplainer(ExplainerModel):
    def explain(self):
        """
            https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
            https://medium.com/analytics-vidhya/interpretability-in-machine-learning-f79e1da4f797
        """
        if not is_regression_by_config(self.cfg):
            forest = RandomForestClassifier()
        else:
            forest = RandomForestRegressor()
 
        forest.fit(self.xts, self.yts)
        return pd.DataFrame({FEATURE: self.id_list, ATTR: forest.feature_importances_})

    def plot(self, df, method=None):
        Graphics().graphic_pie(df, self.prefix + '_RFPermutationImportance_pie.png', 'RF Permutation Feature Importance')
        Graphics().graph_hist(df, self.prefix + '_RFPermutationImportance_hist.png', 'RF Permutation Feature Importance')

