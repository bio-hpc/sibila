#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from ConsensusBase import ConsensusBase

class ConsensusAverageRank(ConsensusBase):

    def __init__(self, folder):
        super(ConsensusAverageRank, self).__init__(folder)
        self.title = 'Average ranking'

    def consensus(self):
        print("Computing average ranking")

        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR, self.RANKING]]

        # average mean of the positions
        df_mean = df.groupby([self.FEATURE])[self.RANKING].mean().to_frame().reset_index()
        df_mean.columns = [self.FEATURE, self.ATTR]    
        df_mean = df_mean.sort_values(self.ATTR)

        # output
        #features = df_mean[self.FEATURE].to_numpy()
        #attrs = df_mean[self.ATTR].to_numpy()
        #return features, attrs
        return df_mean
