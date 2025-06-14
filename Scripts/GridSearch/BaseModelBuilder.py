#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BaseModelBuilder.py:
    Base template for model building

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"

import abc
import itertools as it
import json
import os
from benedict import benedict


class BaseModelBuilder():

    def __init__(self, model_name, directory):
        self.model_name = model_name
        self.directory = directory        

    @staticmethod
    def get_model_list():
        return ['ANN', 'DT', 'KNN', 'RF', 'RLF', 'RNN', 'RP', 'SVM', 'XGBOOST', 'VOT']

    @abc.abstractmethod
    def get_default_model(self):
        """"""

    def build(self, **variations):
        if self.model_name == None:
            raise Exception("Model type must be defined but it's not")
        elif self.model_name not in self.get_model_list():
            raise TypeError("Model type is not allowed. Try: ", self.get_model_list())

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        configs = self.__get_configs(**variations)
        for c in configs:
            outfile, data = os.path.join(self.directory, c[0]), c[1]
            with open(outfile, 'w') as outfile:
                json.dump(data, outfile, indent=4)


    def __get_configs(self, **variations):
        # calculate all the combinations
        l_variations = sorted(variations)
        combinations = list(it.product(*(variations[v] for v in l_variations)))
        
        default = self.get_default_model().copy()
        configs = []

        j = 1
        for c in combinations:
            # update the given properties
            d = benedict(default.copy())
            for i, v in enumerate(l_variations):
                d[v] = c[i]

            aux = json.loads(d.to_json())
            #filename = self.__get_filename(l_variations, c)
            filename = '{}-{}.json'.format(self.model_name, j)  # TODO improve
            configs.append((filename, aux))
            j += 1

        return configs
