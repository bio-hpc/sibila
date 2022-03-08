#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelBuilderRP.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"


from BaseModelBuilder import BaseModelBuilder

class ModelBuilderRP(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = 'NONE'
        p['type_ml'] = 'classification'
        p['n_jobs'] = 8

        p['params'] = {}

        return p
