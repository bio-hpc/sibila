#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import abc
import glob
import json
import os
import pandas as pd
from matplotlib import pyplot as plt

class ConsensusBase(abc.ABC):

    GLOBALS = ['PermutationImportance', 'RFPermutationImportance']
    LOCALS = ['LIME', 'Shapley/csv', 'IntegratedGradients/csv'] #, 'Anchor', 'Dice']
    FEATURE = 'feature'
    ATTR = 'attribution'
    RANKING = 'ranking'

    def __init__(self, folder, out_folder):
        self.folder = folder
        self.out_folder = out_folder
        self.df_g = None
        self.df_l = None
        self.title = None

    def run(self):
        # identify models in the folder
        models, prefixes = self.__find_models()

        # process each model individually
        for i in range(len(models)):
            # load global explanations
            self.__load_globals(i, prefixes)
            # load local explanations
            self.__load_locals(i, prefixes)
            # call consensus
            df = self.consensus()
            # save data into the output folder
            # TODO
            # plot attributions
            self.plot(models[i], df)

    @abc.abstractmethod
    def consensus(self):
        """
        """

    """ """
    def plot(self, model, df):
        print('Plotting attributions after consensus')
        ax = df.plot.bar(x='feature', y='attribution', rot=45)
        if self.title is not None:
            plt.title(self.title)
        plt.show()

    """ Loads the attributions of the global methods """
    def __load_globals(self, idx, prefixes):
        for g in self.GLOBALS:
            foo = prefixes[idx] + '_' + g + '.csv'
            if not self.__load_file(foo):
                continue
            df = self.__load_csv(foo)
            self.__scaler(df)
            df.sort_values(self.ATTR, ascending=False, inplace=True)
            df.insert(len(df.columns), self.RANKING, range(1, 1 + len(df)))

            self.df_g = self.__append_df(self.df_g, df)

    """ Loads the attributions of the local methods """
    def __load_locals(self, idx, prefixes):
        for l in self.LOCALS:
            folder = os.path.split(prefixes[idx])[0] + '/' + l
            foos = self.__find_files(folder)
            for foo in foos:
                df = self.__load_csv(foo)
                self.__scaler(df)
                df.sort_values(self.ATTR, ascending=False, inplace=True)
                df.insert(len(df.columns), self.RANKING, range(1, 1 + len(df)))

                self.df_l = self.__append_df(self.df_l, df)

    """ Load a file """
    def __load_file(self, path):
        return os.path.isfile(path)

    """ Finds those files in a folder with a given extension """
    def __find_files(self, folder, ext='csv'):
        return glob.glob(folder + '/*.' + ext)

    """ Concatenates two dataframes """
    def __append_df(self, df_src, df_tar):
        if df_tar is None:
            return df_src
        if df_src is None:
            return df_tar
        return pd.concat([df_src, df_tar], ignore_index=True)

    """ Finds the models in the given folder """
    def __find_models(self):
        files = glob.glob(self.folder + '/*_data.json')
        paths = []
        models = []
        for foo in files:
            with open(foo, 'rb') as f:
                data = json.load(f)
                paths.append(data['Config']['Prefix'])
                models.append(data['Config']['Model_params']['model'])
        return models, paths

    """ Loads a CSV file """
    def __load_csv(self, path):
        return pd.read_csv(path)

    """ Normalize the 'attribution' column in range [0,1] """
    def __scaler(self, df, column='attribution'):
        #df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        df[column] = df[column]/abs(df[column]).max()*1

