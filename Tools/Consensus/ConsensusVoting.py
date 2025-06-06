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
        self.N = 5

    def consensus(self):
        print("Computing feature voting")
      
        # global methods
        dfg = self.df_g[self.df_g[self.RANKING] <= self.N]
        dfg_cnt = dfg[self.FEATURE].value_counts().to_frame().reset_index()
        dfg_cnt.columns = [self.FEATURE, self.ATTR]

        # local methods
        dfl = self.df_l[self.df_l[self.RANKING] <= self.N]
        dfl_cnt = dfl[self.FEATURE].value_counts().to_frame().reset_index()
        dfl_cnt.columns = [self.FEATURE, self.ATTR]
        #dfl_cnt[self.ATTR] = dfl_cnt[self.ATTR] / self.df_l.shape[0] # total count rated by the number of samples

        # add the counters of local and global methods
        df = pd.concat([dfg_cnt, dfl_cnt], ignore_index=True)
        df_mean = df.groupby(self.FEATURE, as_index=False)[self.ATTR].sum()

        #df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        #df = df[[self.FEATURE, self.ATTR, self.RANKING]]
        # voting features
        #df = df[df[self.RANKING] <= 10]
        #df_mean = df[self.FEATURE].value_counts().to_frame().reset_index()
        #df_mean.columns = [self.FEATURE, self.ATTR]

        return df_mean
