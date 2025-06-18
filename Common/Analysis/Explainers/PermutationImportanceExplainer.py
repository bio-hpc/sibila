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
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Tools.ToolsModels import get_explainer_model, is_tf_model, is_rulefit_model, is_multiclass
from Common.Config.ConfigHolder import FEATURE, ATTR, STD


class PermutationImportanceExplainer(ExplainerModel):
    def explain(self):
        """
            https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
            https://medium.com/analytics-vidhya/interpretability-in-machine-learning-f79e1da4f797
        """
        if len(self.id_list) == 1:
            results = Bunch(importances_mean=1.0, importances_std=0.0)
        else:
            if is_regression_by_config(self.cfg):
                scorer = 'r2'
            elif is_multiclass(self.cfg):
                scorer = 'neg_log_loss'
            else:
                scorer = 'roc_auc'

            if is_tf_model(self.model) or is_rulefit_model(self.model):
                my_model = get_explainer_model(self.model, self.estimator, self.yts, self.cfg)
            else:
                my_model = self.model

            results = permutation_importance(
                my_model,
                self.xtr,
                self.ytr,
                scoring=scorer,
                random_state=self.random_state,
                n_jobs=1,  #to parallelise it uses the pickle library and some models are not compatible, it is left without parallelisation. 
            )

        # build the same structure as the other algorithms
        return pd.DataFrame({FEATURE: self.id_list, ATTR: results.importances_mean, STD: results.importances_std})

    def plot(self, df, method=None):
        Graphics().graphic_pie(df, self.prefix + '_PermutationImportance_pie.png', 'Permutation Feature Importance')
        Graphics().graph_hist(df, self.prefix + '_PermutationImportance_hist.png', 'Permutation Feature Importance')

