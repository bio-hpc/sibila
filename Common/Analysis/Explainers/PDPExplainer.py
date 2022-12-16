#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from Tools.Graphics import Graphics
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Tools.ToolsModels import get_explainer_model


class PDPExplainer(ExplainerModel):
    def explain(self):
        # https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence-plots
        return self.execute()

    def execute(self):
        self.model_ = get_explainer_model(self.model, self.estimator, self.yts, self.cfg)
        if 'XGBRegressor' in str(self.model):
            self.model_.fit(self.xts, self.yts)
            
        xtr_df = pd.DataFrame(data=self.xtr, columns=self.id_list)
        return xtr_df

    def plot(self, df, method=None):
        for i in tqdm(range(len(self.id_list))):
            feat = self.id_list[i]
            pdp_file = '{}{}_PDP_{}.png'.format(
                self.io_data.get_pdp_folder(),
                Path(self.cfg.get_prefix()).stem,
                self.io_data.fix_filename(feat)  # / represents a path in UNIX and breaks the filename
            )
            try:
                Graphics().plot_pdp_ice(self.model_, df, feat, pdp_file, jobs=self.cfg.get_cores(), seed=self.random_state)
            except:
                pass
