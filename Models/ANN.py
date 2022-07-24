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
import keras_tuner as kt


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
                                  self.cfg.get_params()['params_grid']['epochs'],
                                  self.cfg.get_params()['params_grid']['min_lr']))

    def grid_search(self, xtr, ytr):

        params = self.cfg.get_params()['params_grid']
        seed = self.cfg.get_params()['params']['random_state']

        def build_model(hp):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=xtr.shape))
            for i in range(hp.Int("num_layers", 1, params['max_layers'])):
                model.add(
                    tf.keras.layers.Dense(
                        # Tune number of units separately
                        units = hp.Int(f"units_{i}", min_value=params['min_units'], max_value=params['max_units'], step=params['step_units']),
                        activation = hp.Choice("activation", params['activation']) if ('activation' in params and len(params['activation']) > 0) else None,
                        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=seed),
			kernel_regularizer = hp.Choice('kernel_regularizer', params['kernel_regularizer']) if ('kernel_regularizer' in params and len(params['kernel_regularizer']) > 0) else None
                    )
                )

                if hp.Boolean("dropout", params['dropout']):
                    model.add(tf.keras.layers.Dropout(rate=params['dropout_rate']))

                if not is_regression_by_config(self.cfg):
                    model.add(tf.keras.layers.Dense(params['output_units'], activation="softmax", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
                else:
                    model.add(tf.keras.layers.Dense(params['output_units'], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))

                learning_rate = hp.Float("lr", min_value=params['min_lr'], max_value=params['max_lr'], sampling=params['sampling_lr'])
                model.compile(
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss = params["loss_function"],
                    metrics = params['metrics']
                )
            return model

        build_model(kt.HyperParameters())

        tuner = kt.Hyperband(
            hypermodel = build_model,
            objective = kt.HyperParameters().Choice('objective',params["objective"]),
            executions_per_trial = params['executions_per_trial'],
            overwrite = True,
            seed = seed,
            directory = self.cfg.get_folder() + "/keras_tuner_dir",
            project_name = "sibila"
        )

        tuner.search(xtr, ytr, verbose=2, epochs=params['epochs'], callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=params['early_stopping_patience'])])
        return tuner.get_best_hyperparameters(num_trials=1)[0], tuner.get_best_models()[0]

    def cross_validation(self, xtr, ytr):
        params = self.cfg.get_params()['params']
        params_grid = self.cfg.get_params()['params_grid']
        metrics = [ ClassFactory(m).get() for m in params['cv_metrics'] ]
        self.model.compile(loss=params_grid['loss_function'], metrics=metrics)

        if self.cfg.get_args()['crossvalidation'] == None: #or 'cv_splits' not in params:
            for i in range(params_grid['epochs']):
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

    def train(self, xtr, ytr):
        bestHP, self.model = self.grid_search(xtr, ytr)
        #self.cross_validation(xtr, ytr)
        # append the grid search hyperparams to the output
        #params = { **self.cfg.get_params()['params_grid'], **bestHP.values }
        #self.cfg.set_grid_params(params)
        #print('train', xtr.shape, ytr.shape)
        
    def cv_step_fn(self, xtr, ytr, xts=None, yts=None):
        for i in range(self.cfg.get_params()['params_grid']['epochs']):
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

