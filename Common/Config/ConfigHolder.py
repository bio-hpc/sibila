#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ConfigHolder.py:
    Saves all the configuration parameters of the experiment

"""
__author__ = "Jorge de la Peña García"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jorge.dlpg@gmail.com"
__status__ = "Production"

import sys
import time
from os.path import splitext, basename
import git
MAX_SIZE_FEATURES = 100  # maximum features(cols) for Graphics correlation and generate permutations
MAX_IMPORTANCES = 10
CORR_CUTOFF = 0.9


class ConfigHolder:
    F_TEXT = '\t{:>35}:\t{:<10}'
    F_FILE_JSON = '{}_data.json'
    F_RESUME = '{}_resume.txt'
    F_MATRIX_CONFUSION = '{}_confusion_matrix.png'
    F_MATRIX_CORRELATION = '{}_correlation_matrix.png'
    F_ROC_CURVE = '{}_roc.png'
    F_CORRELATION = '{}_correlation.png'
    F_INTERPRETABILITY_TIMES = '{}_interpretability_times.png'
    N_CORES = 8

    def __init__(self, f_dataset, folder, args, params=None):
        """
       :param f_dataset [str]: file dataset
       :param folder [str]: folder test
       :param prefix [str]: prefix to store files
       :param params [{}]: params  dictionary with the parameters and name of the model, for example
                            {
                                "model": "DT",
                                "params": {
                                    "max_leaf_nodes": 10,
                                    "random_state": 20
                            }

       """

        command = ' '.join(sys.argv)
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        #_args = vars(args) if args else {}
        self.config = {
            'Args': vars(args),
            'Command': command,
            'Dataset': f_dataset,
            'Folder': folder,
            'T_ini_ms': time.time(),
            'T_end_ms': '',
            'Time_s': '',
            'Git_version': sha,
            'Model_params': params,
        }
        self.ini_params = params.copy()

    def get_args(self):
        return self.config['Args']

    def get_name_dataset(self):
        return splitext(basename(self.get_dataset()))[0]

    def get_dataset(self):
        return self.config['Dataset']

    def get_params(self):
        return self.config['Model_params']

    def get_name_file_json(self):
        return self.F_FILE_JSON.format(self.config['Prefix'])

    def get_name_file_resume(self):
        return self.F_RESUME.format(self.config['Prefix'])

    def get_name_file_roc(self):
        return self.F_ROC_CURVE.format(self.config['Prefix'])

    def get_name_file_matrix_confusion(self):
        return self.F_MATRIX_CONFUSION.format(self.config['Prefix'])

    def get_name_file_matrix_correlation(self):
        return self.F_MATRIX_CORRELATION.format(self.config['Prefix'])

    def get_name_file_correlation(self):
        return self.F_CORRELATION.format(self.config['Prefix'])

    def get_name_file_interpretability_times(self):
        return self.F_INTERPRETABILITY_TIMES.format(self.config['Prefix'])

    def get_folder(self):
        return self.config['Folder']

    def get_config(self):
        """
            retun json text
        """
        return self.config

    def set_model_params(self, params):
        self.config["Model_params"]["params"] = params

    def get_prefix(self):
        return self.config['Prefix']

    def set_prefix(self, prefix):
        self.config['Prefix'] = prefix

    def set_time_end(self):
        self.config['T_end_ms'] = time.time()
        self.config['Time_s'] = round(self.config['T_end_ms'] - self.config['T_ini_ms'], 3)

    def print_data(self):
        for k, v in self.config.items():
            self.print_format(k, v)

    def set_cores(self, cores):
        self.N_CORES = cores            


    def get_cores(self):
        if self.ini_params:
            return (self._get_cores(self.ini_params))
        else:
            return self.N_CORES

    def _get_cores(self, dct_params):
        N_JOB = "n_job"

        if N_JOB in dct_params.keys():
            return dct_params[N_JOB]
        else:
            for k, v in dct_params.items():
                if N_JOB == k:
                    return v
                elif isinstance(k, str) and N_JOB in k:
                    return v
                elif isinstance(v, dict):
                    return self._get_cores(v)

        return self.N_CORES

    """
        Metodos duplicados en EVALUATIONMETERICS    
    """

    def print_format(self, k, v):
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 == 'Parameters':
                    self.print_m(k2, None)
                elif k2 == 'params_grid':
                    self.print_format(k2, None)
                self.print_format(k2, v2)
        elif isinstance(v, list):
            self.print_format(k, str(v))

        else:
            if v == None: v = 'None'
            self.print_m(k, v)

    def print_m(self, txt, value):
        t = self.F_TEXT.format(txt, str(value)) if (value != None) else "\t{}".format(txt)
        print(t)
        # if self.cfg:
        f = open(self.F_RESUME.format(self.get_prefix()), 'a')  # guardar en fichero
        f.write('{}\n'.format(t))
        f.close()
