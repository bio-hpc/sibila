#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelBuilderSVM.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"


from BaseModelBuilder import BaseModelBuilder

class ModelBuilderSVM(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = 'train_random'
        p['type_ml'] = 'classification'

        p['params'] = {}
        p['params']['kernel'] = 'linear'
        p['params']['random_state'] = 500
        p['params']['probability'] = 1
        p['params']['verbose'] = 1

        return p
