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
import pandas as pd

class ConsensusBase(abc.ABC):

    GLOBALS = ['PermutationImportance', 'RFPermutationImportance'] #, 'PDP', 'ALE']
    LOCALS = ['Anchor', 'LIME', 'Shapley', 'IntegratedGradients', 'Dice']

    def __init__(self, folder):
        self.folder = folder


    def run(self):
        # identify models in the folder
        models, prefixes = self.__find_models()

        # process each model individually
        for i in range(len(models)):
            # load global explanations
            for g in self.GLOBALS:
                df = self.__load_csv(prefixes[i] + '_' + g + '.csv')
                print(df)
            print()

        # TODO cargar algoritmos por columna
        # TODO cargar algoritmos locales
        # TODO aplicar funcion
        pass


    @abc.abstractmethod
    def consensus(self):
        """
        """

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


    def __load_csv(self, path):
        return pd.read_csv(path)

