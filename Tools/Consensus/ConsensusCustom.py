#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from ConsensusBase import ConsensusBase

class ConsensusCustom(ConsensusBase):

    def __init__(self, folder, dir_out):
        super(ConsensusCustom, self).__init__(folder, dir_out)
        self.title = 'Consensus'

    def consensus(self):
        print("Computing custom function")

        if not self.is_regression:
            alfa = (4*pow(self.model_acc, 2))-(4*self.model_acc)+1 # model_acc = AUC
        else:
            alfa = self.model_acc # model_acc = R2

        # global methods
        dfg = self.__scale(self.df_g)
        dfg[self.ATTR] = dfg[self.ATTR] * alfa

        # local methods
        dfl = self.__scale(self.df_l)
        dfl[self.ATTR] = dfl[self.ATTR]/self.df_l.shape[0] # divide by the number of samples to compare with globals
        dfl[self.ATTR] = dfl[self.ATTR] * alfa
        dfl[self.ATTR] = dfl[self.ATTR] * dfl[self.PROBA]

        # merge both transformed explanations
        df = pd.concat([dfg, dfl], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR]]

        # average mean of the attributions
        df_mean = df.groupby([self.FEATURE])[self.ATTR].sum().to_frame().reset_index()
        df_mean = df_mean.reindex(df_mean[self.ATTR].abs().sort_values(ascending=False).index)

        return df_mean

    def __scale(self, df):
        sign = list(map(lambda x: -1 if x<0 else 1, df[self.ATTR].to_numpy()))
        aux = df.copy()
        aux[self.ATTR] = ((df[self.ATTR]-df[self.ATTR].min())/(df[self.ATTR].max()-df[self.ATTR].min()))*sign
        return aux

