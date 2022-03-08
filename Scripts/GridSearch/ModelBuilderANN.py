#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelBuilderANN.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"


from BaseModelBuilder import BaseModelBuilder

class ModelBuilderANN(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['type_ml'] = 'classification'
        p['n_job'] = 1
        p['params'] = {}
        p['params']['draw_model'] = False
        p['params']['random_state'] = 500
        p['params']['batch_size'] = 1024

        p['params']['epochs'] = 200
        p['params']['loss_function'] = 'binary_crossentropy'
        p['params']['cv_splits'] = 3

        p['params']['optimizer'] = {}
        p['params']['optimizer']['type'] = 'tensorflow.keras.optimizers.Adam'
        p['params']['optimizer']['properties'] = {}
        p['params']['optimizer']['properties']['learning_rate'] = 0.01
        p['params']['optimizer']['properties']['beta_1'] = 0.99
        p['params']['optimizer']['properties']['beta_2'] = 0.999
        p['params']['optimizer']['properties']['epsilon'] = 1e-8
        
        p['params']['layers'] = {}
        p['params']['layers']['dense_0'] = {}
        p['params']['layers']['dense_0']['type'] = 'tensorflow.keras.layers.Dense'
        p['params']['layers']['dense_0']['properties'] = {}
        p['params']['layers']['dense_0']['properties']['units'] = 16
        p['params']['layers']['dense_0']['properties']['activation'] = 'relu'
        p['params']['layers']['dense_0']['properties']['name'] = 'dense_0'
        
        p['params']['layers']['dense_1'] = {}
        p['params']['layers']['dense_1']['type'] = 'tensorflow.keras.layers.Dense'
        p['params']['layers']['dense_1']['properties'] = {}
        p['params']['layers']['dense_1']['properties']['units'] = 8
        p['params']['layers']['dense_1']['properties']['activation'] = 'relu'
        p['params']['layers']['dense_1']['properties']['name'] = 'dense_1'
        
        p['params']['layers']['dense_2'] = {}
        p['params']['layers']['dense_2']['type'] = 'tensorflow.keras.layers.Dense'
        p['params']['layers']['dense_2']['properties'] = {}
        p['params']['layers']['dense_2']['properties']['units'] = 1
        p['params']['layers']['dense_2']['properties']['activation'] = 'sigmoid'
        p['params']['layers']['dense_2']['properties']['name'] = 'dense_2'

        p['params']['metrics'] = [ 
            'tensorflow.keras.metrics.MeanAbsoluteError', 
            'tensorflow.keras.metrics.MeanSquaredError', 
            'tensorflow.keras.metrics.MeanSquaredLogarithmicError', 
            'tensorflow.keras.metrics.RootMeanSquaredError' 
        ]
        return p
