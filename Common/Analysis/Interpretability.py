#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interpretability.py:
https://medium.com/analytics-vidhya/interpretability-in-machine-learning-f79e1da4f797

"""
__author__ = "Jorge de la Peña García"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jorge.dlpg@gmail.com"
__status__ = "Production"

import os.path
import sys
import pandas as pd
import time
from Common.Config.ConfigHolder import MAX_IMPORTANCES
from Tools.PostProcessing.Serialize import Serialize
from Tools.IOData import serilize_class
from Tools.IOData import get_serializa_params
from Common.Analysis.Explainers import *
from Tools.Timer import Timer
from Tools.Graphics import Graphics
from os.path import basename, dirname, normpath
from glob import glob


class Interpretability:
    # FeatureImportance only works with DT, RF, SVM and KNN
    TEST_METHODS = []
    PARALLEL_METHODS = ['Lime', "Shapley", "IntegratedGradients", 'Dice', 'PDP']
    COMMON_METHODS = ['PermutationImportance', 'ALE']
    METHODS = {
        "DT": [],
        "RF": [],
        "SVM": [],
        "XGBOOST": [],
        "ANN": [],
        "KNN": [],
        "RP": [],
        "RLF": []
    }

    def __init__(self, serialize_params):
        """

        """
        params = serialize_params.get_params()
        run_method = params['run_method']
        del params['run_method']
        if run_method:
            self.execute_method(params, run_method)
        else:
            self.execute(params)

        name_model = params['cfg'].get_params()['model']
        self.plot_times(params['cfg'].get_folder(), params['cfg'].get_prefix(), name_model)

    def execute(self, params):

        if len(self.TEST_METHODS) > 0:
            self.execute_methods(params, self.TEST_METHODS)
        else:
            self.execute_methods_parellel(params, self.PARALLEL_METHODS)
            self.execute_methods(params, self.COMMON_METHODS)
            name_model = params['cfg'].get_params()['model'].upper()
            self.execute_methods(params, self.METHODS[name_model])

    def execute_methods(self, params, lst_methods):
        for method in lst_methods:
            self.execute_method(params, method)

    def execute_methods_parellel(self, params, lst_method):
        for method in lst_method:
            if not params['cfg'].get_args()['queue']:
                self.execute_method(params, method)
            else:  # if parallelism is required, the test data is serialised
                self.serialize_params(params)
                #print("python3 -m Common.Analysis.Interpretability {} {}".format(params['cfg'].get_prefix() + '.pkl',
                #                                                                 method))

    def execute_method(self, params, method):
        t = Timer(method)
        obj = globals()[method + 'Explainer'](**params)
        df = obj.explain()
        if df is not None and 'PDP' not in method:
            df = df.reindex(df['weight'].abs().sort_values(ascending=False).index)
            if len(params['id_list']) > MAX_IMPORTANCES:
                n_others = len(params['id_list']) - MAX_IMPORTANCES
                title = 'Sum other {} features'.format(str(n_others))
                df_others = pd.DataFrame(data=[[title, df[MAX_IMPORTANCES:]['weight'].sum()]],
                                         columns=['feature', 'weight'])
                df = pd.concat([df[:MAX_IMPORTANCES], df_others], ignore_index=True)

        if df is not None:
            obj.plot(df, method=method)

        file_time = '{}_{}_time.txt'.format(params['cfg'].get_prefix(), method)
        t.save(file_time, params['io_data'])
        self.print_data(t.total(), method, params['io_data'])

    def serialize_params(self, params):
        class_serializer = Serialize(**params)
        if not os.path.isfile(params['cfg'].get_prefix() + '_params.pkl'):
            serilize_class(class_serializer, params['cfg'].get_prefix() + '_params.pkl')

    def print_data(self, total_time, method, io_data):
        io_data.print_m("{}: Total time: {} s".format(method, round(total_time, 3)))

    def plot_times(self, dir_name, prefix, name_model):
        dct_times = {}
        lst_files = glob('{}*_time.txt'.format(prefix)) + [os.path.join(dir_name, 'load_time.txt')]

        for file in lst_files:
            with open(file) as f:
                first_line = f.readline()
            method = first_line.split(":")[0]
            time = round(float(first_line.split(":")[1].strip()), 3)
            position = round(float(first_line.split(":")[2].strip()))

            dct_times[method] = (time, position)
        # sort entries by position: entry=(time, position)
        dct_times = dict(sorted(dct_times.items(), key=lambda x: x[1][1]))

        Graphics().plot_interpretability_times(dct_times, prefix + "_times.png", name_model)


if __name__ == "__main__":
    serialize_file = sys.argv[1]
    method = sys.argv[2]
    cl_serialize = get_serializa_params(serialize_file)
    cl_serialize.set_run_method(method)
    Interpretability(cl_serialize)
