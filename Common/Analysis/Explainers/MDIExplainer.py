#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from Tools.ToolsModels import is_regression_by_args
from Tools.Graphics import Graphics
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Common.Config.ConfigHolder import FEATURE, ATTR, STD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MDIExplainer(ExplainerModel):
    EXPLAINER_NAME = 'MDI'

    def explain(self):
        """
            https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
            https://medium.com/analytics-vidhya/interpretability-in-machine-learning-f79e1da4f797
        """
        mdi_cfg = self.get_explainer_cfg()
        random_state = mdi_cfg['random_state']
        if random_state is None:
            random_state = self.random_state

        if not is_regression_by_args(self.cfg.get_args()):
            forest = RandomForestClassifier(
                n_estimators=mdi_cfg['n_estimators'],
                random_state=random_state
            )
        else:
            forest = RandomForestRegressor(
                n_estimators=mdi_cfg['n_estimators'],
                random_state=random_state
            )
 
        forest.fit(self.xts, self.yts)
        return pd.DataFrame({FEATURE: self.id_list, ATTR: forest.feature_importances_})

    def plot(self, df, method=None):
        Graphics().graphic_pie(df, self.prefix + '_MDI_pie.png', 'Mean Decrease in Impurity')
        Graphics().graph_hist(df, self.prefix + '_MDI_hist.png', 'Mean Decrease in Impurity')

