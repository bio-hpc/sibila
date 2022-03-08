#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelBuilderRF.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"


from BaseModelBuilder import BaseModelBuilder

class ModelBuilderRF(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = 'None'
        p['type_ml'] = 'classification'

        p['params'] = {}
        p['params']['n_estimators'] = 100
        p['params']['criterion'] = 'gini'
        p['params']['max_depth'] = None
        p['params']['min_samples_split'] = 2
        p['params']['bootstrap'] = False
        p['params']['max_features'] = 'sqrt'
        p['params']['verbose'] = 1
        p['params']['n_jobs'] = 4
        p['params']['min_weight_fraction_leaf'] = 0

        return p
