#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interpretability.py:
https://medium.com/analytics-vidhya/interpretability-in-machine-learning-f79e1da4f797

"""

import sys
import pandas as pd
import time
from Common.Config.ConfigHolder import ATTR, FEATURE, MAX_IMPORTANCES
from Tools.IOData import get_serialized_params
from Common.Analysis.Explainers import *
from Tools.Timer import Timer
from Tools.Graphics import Graphics
from os.path import basename, dirname, normpath
from Tools.Bash.Queue_manager.JobManager import JobManager
from Tools.Bash.Queue_manager.jobs import get_nitems_per_block
from Tools.ToolsModels import is_multiclass

class Interpretability:
    # FeatureImportance only works with DT, RF, SVM and KNN
    TEST_METHODS = []
    PARALLEL_METHODS = ['PermutationImportance', 'RFPermutationImportance', 'Lime', 'Shapley', 'IntegratedGradients', 'Dice', 'Anchor', 'PDP', 'ALE']
    COMMON_METHODS = []
    METHODS = {
        "DT": [],
        "RF": [],
        "SVM": [],
        "XGBOOST": [],
        "ANN": [],
        "KNN": [],
        "RP": [],
        "RLF": [],
        "LR": [],
        "BAG": []
    }

    def __init__(self, serialize_params, block_nr=None):
        self.block_nr = block_nr
        params = serialize_params.get_params()
        run_method = params['run_method']
        del params['run_method']
        if run_method:
            self.execute_method(params, run_method)
        else:
            self.execute(params)

    def execute(self, params):
        if len(self.TEST_METHODS) > 0:
            self.execute_methods(params, self.TEST_METHODS)
        else:
            self.execute_methods_parallel(params, self.PARALLEL_METHODS)
            self.execute_methods(params, self.COMMON_METHODS)
            name_model = params['cfg'].get_params()['model'].upper()
            self.execute_methods(params, self.METHODS[name_model])

    def execute_methods(self, params, lst_methods):
        for method in lst_methods:
            self.execute_method(params, method)

    def execute_methods_parallel(self, params, lst_method):
        if params['cfg'].get_args()['queue']:
            jm = JobManager()
            jm.parallelize(params, lst_method)
        else:
            for method in lst_method:
                self.execute_method(params, method)

    def execute_method(self, params, method):
        # when a block number is given, only that part of the data is taken
        if not self.block_nr is None:
            N = get_nitems_per_block(method, params['xts'].shape)
            xts_ith, yts_ith, idx_ith = self.take_data(params['xts'], params['yts'], params['idx_xts'], int(self.block_nr), N)

            new_params = params.copy()
            new_params['xts'] = xts_ith
            new_params['yts'] = yts_ith
            new_params['idx_xts'] = idx_ith
        else:
            new_params = params.copy()

        t = Timer(method)
        obj = globals()[method + 'Explainer'](**new_params)
        df = obj.explain()

        if df is not None:
            if ATTR in df.columns:
                df = self.sort(df)
            params['io_data'].save_dataframe_cols(df, df.columns, params['cfg'].get_prefix()+'_'+method+'.csv')
            df = self.shorten_features(df, method, len(new_params['id_list']))
            obj.plot(df, method=method)

        file_time = '{}_{}_time.txt'.format(new_params['cfg'].get_prefix(), method)
        t.save(file_time, new_params['io_data'])
        self.print_data(t.total(), method, new_params['io_data'])

    def print_data(self, total_time, method, io_data):
        io_data.print_m("{}: Total time: {} s".format(method, round(total_time, 3)))

    def take_data(self, xts, yts, idx, block_id, N):
        xts_splited = [xts[x:x+N] for x in range(0, len(xts), N)]
        yts_splited = [yts[x:x+N] for x in range(0, len(yts), N)]
        idx_splited = [idx[x:x+N] for x in range(0, len(idx), N)]
        return xts_splited[block_id], yts_splited[block_id], idx_splited[block_id]

    def shorten_features(self, df, method, n_features):
        if df is not None and 'PDP' not in method:
            df = df.reindex(df[ATTR].abs().sort_values(ascending=False).index)
            if n_features > MAX_IMPORTANCES:
                n_others = n_features - MAX_IMPORTANCES
                title = 'Sum other {} features'.format(str(n_others))
                df_others = pd.DataFrame(data=[[title, df[MAX_IMPORTANCES:][ATTR].sum()]],
                                         columns=[FEATURE, ATTR])
                df = pd.concat([df[:MAX_IMPORTANCES], df_others], ignore_index=True)

        return df

    def sort(self, df):
        return df.reindex(df[ATTR].abs().sort_values(ascending=False).index)


if __name__ == "__main__":
    serialize_file = sys.argv[1]
    method = sys.argv[2]
    idx = sys.argv[3]

    cl_serialize = get_serialized_params(serialize_file)
    cl_serialize.set_run_method(method)
    Interpretability(cl_serialize, idx)

