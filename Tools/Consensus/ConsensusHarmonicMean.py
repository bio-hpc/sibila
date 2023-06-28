#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from ConsensusBase import ConsensusBase
from scipy.stats.mstats import hmean

class ConsensusHarmonicMean(ConsensusBase):

    def __init__(self, folder, dir_out):
        super(ConsensusHarmonicMean, self).__init__(folder, dir_out)
        self.title = 'Harmonic mean'

    def consensus(self):
        print("Computing harmonic mean")

        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR]]

        # re-scale to [0,1]
        df[self.ATTR] = (df[self.ATTR] - df[self.ATTR].min()) / (df[self.ATTR].max() - df[self.ATTR].min())    
        #df[self.ATTR] = abs(df[self.ATTR]) # absolute value of the attribution

        # geometric mean of the attributions
        df_mean = df.groupby([self.FEATURE]).apply(hmean).reset_index()
        df_mean.columns = [self.FEATURE, self.ATTR]
        df_mean[self.ATTR] = df_mean[self.ATTR].astype(float)

        # output
        #features = df_mean[self.FEATURE].to_numpy()
        #attrs = df_mean[self.ATTR].to_numpy()
        #return features, attrs
        return df_mean
