#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
import numpy as np
from Tools.Graphics import Graphics
from Tools.ToolsModels import is_tf_model, is_rulefit_model, is_regression_by_config
from alibi.explainers import ALE
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Common.Config.ConfigHolder import FEATURE, ATTR

class ALEExplainer(ExplainerModel):
    def explain(self):
        # https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.ale.html
        targets = np.unique(self.yts).astype(str)

        if is_regression_by_config(self.cfg):
            ale = ALE(self.model.predict, feature_names=self.id_list)
        elif is_tf_model(self.model):
            ale = ALE(self.model.predict, feature_names=self.id_list, target_names=targets)
        else:
            ale = ALE(self.model.predict_proba, feature_names=self.id_list, target_names=targets)

        self.exp = ale.explain(self.xts)
        #self.exp.feature_names
        return pd.DataFrame({FEATURE: [], ATTR: []})

    def plot(self, df, method=None):
        for i in tqdm(range(len(self.id_list))):
            feat = self.id_list[i]

            ale_file = '{}{}_ALE_{}.png'.format(self.io_data.get_ale_folder(),
                                                Path(self.cfg.get_prefix()).stem, self.io_data.fix_filename(feat))
            Graphics().plot_ale(self.exp, feat, ale_file)

