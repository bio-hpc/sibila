#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from ConsensusBase import ConsensusBase
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np


"""
WSCF: Weighted and Scaled Consensus Function
"""
class ConsensusWSCF(ConsensusBase):

    #WEIGHTS = {
    #    "Shapley/csv": 0.350,
    #    "RFPermutationImportance": 0.250,
    #    "PermutationImportance": 0.190,
    #    "LIME": 0.190,
    #    "IntegratedGradients/csv": 0.010,
    #    "Dice": 0.010
    #}

    def __init__(self, folder, dir_out):
        super(ConsensusWSCF, self).__init__(folder, dir_out)
        self.title = 'WSCF'

    def consensus(self):
        print("Computing WSCF")

        # a) global methods
        # reload the data to manipulate it individually
        models, prefixes = super().find_models()
        self.df_g = None
        scaler = MinMaxScaler(feature_range=(0, 1))

        for idx in range(len(models)):
            for g in self.GLOBALS:
                foo = prefixes[idx] + '_' + g + '.csv'
                if not super().load_file(foo):
                    continue
                df = super().load_csv(foo)

                # scale in range [0,1]
                df[self.ATTR] = scaler.fit_transform(df[[self.ATTR]])
                #df[self.ATTR] *= self.WEIGHTS[g] # TODO
                self.df_g = super().append_df(self.df_g, df)

        self.df_g['std'] = self.df_g['std'].fillna(0)

        # b) local methods
        # reload the data to manipulate individually
        self.df_l = None
        scaler = MinMaxScaler(feature_range=(0, 1))

        for idx in range(len(models)):
            for l in self.LOCALS:
                # Neural networks don't work with counterfactuals
                if models[idx] == 'ANN' and l == 'Dice':
                    continue

                folder = os.path.split(prefixes[idx])[0] + '/' + l
                foos = self.find_files(folder)
                for foo in foos:
                    df = self.load_csv(foo)

                    # scale in range [0,1]
                    df[self.ATTR] = scaler.fit_transform(df[[self.ATTR]])

                    # divide by the number of samples to avoid bias against global methods
                    N = df.shape[0]
                    df[self.ATTR] = df[[self.ATTR]] / N
                    #df[self.ATTR] *= self.WEIGHTS[l] # TODO
                    self.df_l = self.append_df(self.df_l, df)

        # apply correction factor to local methods for classification and regression
        if self.is_regression:
            self.df_l['error'] = self.df_l[self.TRUEVAL] - self.df_l[self.PREDVAL]
            self.df_l['exp_factor'] = self.df_l['error'].apply(lambda x: self._exponential_factor(x, alpha=0.5)/N)
            self.df_l[self.ATTR] = self.df_l[self.ATTR] * self.df_l['exp_factor']           
        else:
            self.df_l[self.ATTR] *= (4 * (self.df_l[self.PROBA]**2 - self.df_l[self.PROBA]) + 1)

        # merge both transformed explanations
        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR]]
        df_acc = df.groupby(self.FEATURE, as_index=False).agg({self.ATTR: "sum"})

        return df_acc

    def _exponential_factor(self, error, alpha=0.5):
        return np.exp(-alpha * abs(error))

    def _sigmoid_factor(self, error, beta=0.5):
        return 1 / (1 + beta * abs(error))
