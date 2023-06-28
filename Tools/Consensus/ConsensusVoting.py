#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from ConsensusBase import ConsensusBase

class ConsensusVoting(ConsensusBase):

    def __init__(self, folder, dir_out):
        super(ConsensusVoting, self).__init__(folder, dir_out)
        self.title = 'Voting'

    def consensus(self):
        print("Computing feature voting")

        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR, self.RANKING]]

        # voting features
        df = df[df[self.RANKING] <= 10]
        df_mean = df[self.FEATURE].value_counts().to_frame().reset_index()
        df_mean.columns = [self.FEATURE, self.ATTR]

        # output
        #features = df_mean[self.FEATURE].to_numpy()
        #attrs = df_mean[self.ATTR].to_numpy()
        #return features, attrs
        return df_mean
