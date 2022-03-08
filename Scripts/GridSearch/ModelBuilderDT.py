#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelBuilderDT.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"


from BaseModelBuilder import BaseModelBuilder

class ModelBuilderDT(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = 'train_random'
        p['type_ml'] = 'classification'
        
        p['params'] = {}
        p['params']['max_leaf_nodes'] = 10

        return p
