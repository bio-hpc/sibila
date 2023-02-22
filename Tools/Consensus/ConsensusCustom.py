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
        self.title = 'Custom function'

    def consensus(self):
        print("Computing custom function")

        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        #df = df[[self.FEATURE, self.ATTR]]

        print(df)
        exit()

        # average mean of the attributions
        #df_mean = df.groupby([self.FEATURE])[self.ATTR].mean().to_frame().reset_index()
        #df_mean = df_mean.reindex(df_mean[self.ATTR].abs().sort_values(ascending=False).index)
        
        # output
        #features = df_mean[self.FEATURE].to_numpy()
        #attrs = df_mean[self.ATTR].to_numpy()
        #return features, attrs
        return df_mean
