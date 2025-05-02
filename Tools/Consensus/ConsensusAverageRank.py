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

    def __init__(self, folder, dir_out):
        super(ConsensusAverageRank, self).__init__(folder, dir_out)
        self.title = 'Average ranking'

    def consensus(self):
        print("Computing average ranking")

        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR, self.RANKING]]

        # average the positions
        df_mean = df.groupby([self.FEATURE])[self.RANKING].mean().to_frame().reset_index()
        df_mean.columns = [self.FEATURE, self.ATTR]

        # reverse the order to turn smaller positions into the most attributed ones 
        df_mean[self.ATTR] = 1 / df_mean[self.ATTR]
        df_mean = df_mean.sort_values(self.ATTR, ascending=False)

        # output
        return df_mean
