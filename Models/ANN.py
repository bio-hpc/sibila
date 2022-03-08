#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANN.py:
    Implementation of Artificial Neural Network (ANN) models

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

from .BaseModel import BaseModel
from .Utils.CrossValidation import CrossValidation
from Tools.ClassFactory import ClassFactory
from Tools.Graphics import Graphics
from os.path import basename, join, splitext
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from alibi.explainers import IntegratedGradients
from Tools.ToolsModels import is_regression_by_config, make_model

PREFIX_OUT_ANN = '{}_{}_{}_{}'  # Model, Dataset, Epochs, Learning rate


class ANN(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(ANN, self).__init__(io_data, cfg, id_list)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = make_model(cfg, id_list)
        self.graphics = Graphics()
        self.history = []

    def get_prefix(self):
        if self.cfg.get_args()['parameters'] is not None:
            foo = self.cfg.get_args()['parameters'][0]
            model_name = splitext(basename(foo))[0]
        else:
            model_name = self.cfg.get_params()['model']

        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_ANN.format(model_name, self.cfg.get_name_dataset(),
                                  self.cfg.get_params()['params']['epochs'],
                                  self.cfg.get_params()['params']['optimizer']['properties']['learning_rate']))

    def train(self, xtr, ytr):
        params = self.cfg.get_params()['params']
        metrics = [ClassFactory(m).get() for m in params['metrics']]
        optimizer = ClassFactory(params['optimizer']['type'], **params['optimizer']['properties']).get()

        self.model.compile(optimizer=optimizer, loss=params['loss_function'], metrics=metrics)

        if self.cfg.get_args()['crossvalidation'] == None or 'cv_splits' not in params:
            for i in range(self.cfg.get_params()['params']['epochs']):
                self.history.append(self.model_fit(xtr, ytr))
        else:
            # cope with different parameters in each method
            cv_params = {
                'n_splits': params.get('cv_splits', CrossValidation.N_SPLITS),
                'random_state': params.get('random_state', CrossValidation.RANDOM_STATE)
            }

            cv = CrossValidation(self.io_data)
            method = cv.choice_method(self.cfg.get_args()['crossvalidation'])
            cv.run_method(method, xtr, ytr, self.cv_step_fn, **cv_params)

    def cv_step_fn(self, xtr, ytr, xts=None, yts=None):
        for i in range(self.cfg.get_params()['params']['epochs']):
            self.history.append(self.model_fit(xtr, ytr))

    def predict(self, xts):  # Make a prediction
        ypr = self.model_predict(xts)

        # draw model and print it on the screen
        self.model.summary()
        self.plot_history()
        if self.cfg.get_params()['params']['draw_model']:
            self.graphics.draw_model(self.model, self.cfg.get_prefix())

        if is_regression_by_config(self.cfg):
            return np.squeeze(ypr)

        return ypr.argmax(axis=1)

    def plot_history(self):
        training_data = [h.history['loss'] for h in self.history]
        file_out = self.get_prefix() + "_lc_tf.png"
        self.graphics.plot_learning_curves(training_data, file_out, metric='Loss')
