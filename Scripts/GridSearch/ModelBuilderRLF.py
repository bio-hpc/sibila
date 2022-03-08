#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelBuilderRLF.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"


from BaseModelBuilder import BaseModelBuilder

class ModelBuilderRLF(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = 'NONE'
        p['type_ml'] = 'classification'
        p['n_jobs'] = 8

        p['params'] = {}
        p['params']['tree_size'] = 4
        p['params']['sample_fract'] = 'default'
        p['params']['max_rules'] = 2000
        p['params']['memory_par'] = 0.01
        p['params']['rfmode'] = 'classify'
        p['params']['lin_trim_quantile'] = 0.025
        p['params']['lin_standardise'] = True
        p['params']['exp_rand_tree_size'] = True

        p['params_grid'] = {}

        return p
