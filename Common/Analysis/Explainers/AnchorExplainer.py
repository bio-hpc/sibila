#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import numpy as np
import pandas as pd
from Tools.Graphics import Graphics
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Common.Config.ConfigHolder import FEATURE, ATTR, STD, PROBA
from alibi.explainers import AnchorTabular
from tqdm import tqdm
from pathlib import Path

class AnchorExplainer(ExplainerModel):
    def explain(self):
        """
            https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
            https://medium.com/analytics-vidhya/interpretability-in-machine-learning-f79e1da4f797
        """
        predict_fn = None
        if hasattr(self.model, 'predict_proba') and callable(self.model.predict_proba):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        #explainer = AnchorTabular(self.model.predict_proba, self.id_list, seed=self.cfg.get_args()['seed'])
        explainer = AnchorTabular(predict_fn, self.id_list, seed=self.cfg.get_args()['seed'])
        explainer.fit(self.xtr, disc_perc=(25, 50, 75))

        df_local = []
        for i in tqdm(range(len(self.xts))):
            explanation = explainer.explain(self.xts[i], threshold=0.95)

            features = np.unique([self.id_list[f] for f in explanation.raw['feature']])
            precisions = [ explanation.precision ]*len(features)
            coverages = [ explanation.coverage ]*len(features)
            rules = [ ' AND '.join(explanation.anchor) ]*len(features)
            proba = self.proba_sample(self.xts[i])

            # local interpretability
            df = pd.DataFrame({FEATURE: features, 'precision': precisions, 'coverage': coverages, 'rule': rules, PROBA: proba})

            prefix = Path(self.cfg.get_prefix()).stem
            out_file = self.io_data.get_anchor_folder() + '{}_Anchor_{}.csv'.format(prefix, self.idx_xts[i])
            self.io_data.save_dataframe_cols(df, df.columns, out_file)

            df_local.append(df)

        # global interpretability
        df_prec = pd.concat(df_local).groupby('rule')['precision'].agg(['mean','std']) # groupby('FEATURE')
        df_cov = pd.concat(df_local).groupby('rule')['coverage'].sum() # groupby('FEATURE')
        df_global = pd.merge(df_prec, df_cov, on='rule').reset_index()
        df_global.columns = [FEATURE, 'precision', 'std', 'coverage']

        out_file = self.cfg.get_prefix() + '_Anchor.csv'
        self.io_data.save_dataframe_cols(df_global, df_global.columns, out_file)
        return None

    def plot(self, df, method=None):
        pass
