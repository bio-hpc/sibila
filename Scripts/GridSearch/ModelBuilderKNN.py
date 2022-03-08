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

class ModelBuilderKNN(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = 'train_random'
        p['type_ml'] = 'classification'

        p['params'] = {}
        p['params']['n_neighbors'] = 5
        p['params']['algorithm'] = 'auto'
        p['params']['leaf_size'] = 30
        p['params']['p'] = 2
        p['params']['metric'] = 'minkowski'
        p['params']['n_jobs'] = -1
        p['params']['weights'] = 'distance'

        return p

