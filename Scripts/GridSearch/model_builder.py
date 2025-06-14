#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_builder.py:
    Creates JSON config files for testing the models
"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"

import argparse
from BaseModelBuilder import BaseModelBuilder
from ModelBuilderANN import ModelBuilderANN
from ModelBuilderDT import ModelBuilderDT
from ModelBuilderKNN import ModelBuilderKNN
from ModelBuilderRF import ModelBuilderRF
from ModelBuilderRLF import ModelBuilderRLF
from ModelBuilderRNN import ModelBuilderRNN
from ModelBuilderRP import ModelBuilderRP
from ModelBuilderSVM import ModelBuilderSVM
from ModelBuilderXGBOOST import ModelBuilderXGBOOST
from ModelBuilderVOT import ModelBuilderVOT

def get_VOT():
    return {
        'params.voting_type': ['majority', 'weighted'],
        'params.base_models': [
            [
                {"model_name": "KNN", "model_class": "ModelBuilderKNN"},
                {"model_name": "DT", "model_class": "ModelBuilderDT"}
            ]
        ],
        'params.weights': [
            {"KNN": 1.0, "DT": 1.0},
            {"KNN": 0.5, "DT": 1.5}
        ]
    }

def get_ANN():
  return {
       ## basics
       'params.epochs': [ 200, 500, 1000 ],
       'params.loss_function': ['sparse_categorical_crossentropy', 'mse', 'mae'],
       'params.cv_splits': [ 2, 5, 10 ],
       ## optimizer
       'params.optimizer': [
           { 'type': 'tensorflow.keras.optimizers.Adam' }, 
           { 'type': 'tensorflow.keras.optimizers.RMSprop' }, 
           { 'type': 'tensorflow.keras.optimizers.SGD' }
       ],
       'params.optimizer.properties.learning_rate': [ .01, .001 ],
       ## network topology
       'params.layers': [
           {
               "dense_0": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 16,
                       "activation": "relu",
                       "name": "dense_0",
                       "kernel_regularizer": "l2"
                   }
               },
               "dense_2": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 16,
                       "activation": "relu",
                       "name": "dense_2",
                       "kernel_regularizer": "l2"
                   }
               },
               "dense_1": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 2,
                       "activation": "softmax",
                       "name": "dense_1"
                   }
               }
           },
           {
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 16,
                      "activation": "relu",
                      "name": "dense_0",
                      "kernel_regularizer": "l2"
                  }
              },
              "dense_2": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "name": "dense_2",
                      "activation": "relu",
                      "kernel_regularizer": "l2"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 2,
                      "activation": "softmax",
                      "name": "dense_1",
                      "kernel_regularizer": "l2"
                  }
              }
           }
      ]
    }

def get_RNN():
  return Models.RNN.name, {
       ## basics
       'params.epochs': [ 100, 200, 500 ],
       'params.loss_function': ['binary_crossentropy', 'mse', 'mae'],
       'params.cv_splits': [ 2, 3, 4, 5 ],
       ## optimizer
       'params.optimizer': [
           { 'type': 'tensorflow.keras.optimizers.Adam' }, 
           { 'type': 'tensorflow.keras.optimizers.RMSprop' }, 
           { 'type': 'tensorflow.keras.optimizers.SGD' }
       ],
       'params.optimizer.properties.learning_rate': [ .01, .001 ],
       ## network topology
       'params.layers': [
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.SimpleRNN",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
               "dense_0": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 8,
                       "activation": "relu",
                       "name": "dense_0"
                   }
               },
               "dense_1": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 1,
                       "activation": "sigmoid",
                       "name": "dense_1"
                   }
               }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.SimpleRNN",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "activation": "relu",
                      "name": "dense_0"
                  }
              },
              "dropout_0": {
                  "type": "tensorflow.keras.layers.Dropout",
                  "properties": {
                      "rate": 0.2,
                      "name": "dropout_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.SimpleRNN",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 4,
                      "activation": "relu",
                      "name": "dense_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.SimpleRNN",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "activation": "sigmoid",
                      "name": "dense_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.LSTM",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
               "dense_0": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 8,
                       "activation": "relu",
                       "name": "dense_0"
                   }
               },
               "dense_1": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 1,
                       "activation": "sigmoid",
                       "name": "dense_1"
                   }
               }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.LSTM",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "activation": "relu",
                      "name": "dense_0"
                  }
              },
              "dropout_0": {
                  "type": "tensorflow.keras.layers.Dropout",
                  "properties": {
                      "rate": 0.2,
                      "name": "dropout_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.LSTM",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 4,
                      "activation": "relu",
                      "name": "dense_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.LSTM",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "activation": "sigmoid",
                      "name": "dense_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.GRU",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
               "dense_0": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 8,
                       "activation": "relu",
                       "name": "dense_0"
                   }
               },
               "dense_1": {
                   "type": "tensorflow.keras.layers.Dense",
                   "properties": {
                       "units": 1,
                       "activation": "sigmoid",
                       "name": "dense_1"
                   }
               }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.GRU",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "activation": "relu",
                      "name": "dense_0"
                  }
              },
              "dropout_0": {
                  "type": "tensorflow.keras.layers.Dropout",
                  "properties": {
                      "rate": 0.2,
                      "name": "dropout_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.GRU",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 4,
                      "activation": "relu",
                      "name": "dense_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           },
           {
               "rnn_0": {
                   "type": "tensorflow.keras.layers.GRU",
                   "properties": {
                       "units": 8,
                       "kernel_initializer": "ones",
                       "name": "rnn_0"
                   }
               },
              "dense_0": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 8,
                      "activation": "sigmoid",
                      "name": "dense_0"
                  }
              },
              "dense_1": {
                  "type": "tensorflow.keras.layers.Dense",
                  "properties": {
                      "units": 1,
                      "activation": "sigmoid",
                      "name": "dense_1"
                  }
              }
           }
      ]
    }

def get_SVM():
  return { 'params.kernel': ['linear', 'poly', 'rbf', 'sigmoid'] }

def get_RF():
  return {
      'params.n_estimators': [50, 100, 400, 800],
      'params.criterion': ['gini', 'entropy'],
      'params.max_depth': [25, 50, 250],
      'params.min_samples_split': [2, 5, 10],
      'params.min_samples_leaf': [2, 5, 10],
      'params.max_features': ['auto', 'sqrt', 'log2'],
      'params.oob_score': [True, False],
      'params.bootstrap': [True, False]
  }

def get_RLF():
  return {}

def get_RP():
  return {}

def get_DT():
  return {
      'params.criterion': ['gini', 'entropy'],
      'params.splitter': ['best', 'random'],
      'params.max_depth': [2, 4, 6, 8, 10, 12],
      'params.min_samples_split': [.1, .2, .4, .8, .9],
      'params.min_samples_leaf': [1, 2, 3, 4],
      'params.min_weight_fraction_leaf': [.1, .2, .3, .4, .5],
      'params.max_features': ['auto', 'sqrt', 'log2'],
      'params.max_leaf_nodes': [5, 10, 15, 20, 30, 40, 50],
      'params.min_impurity_decrease': [0, .1, .2, .3, .4, .5],
      'params.ccp_alpha': [0, .1, .2, .3, .4, .5]
  }

def get_XGBOOST():
  return {
      'params.xgbclassifier__gamma': [.5, 1],
      'params.xgbclassifier__max_depth': [3, 4]
  }

def get_KNN():
  return {
      'params.n_neighbors': [3, 4, 5, 6, 7],
      'params.algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'params.leaf_size': [10, 20, 30, 50],
      'params.metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
      'p': [1, 2, 3]
  }

# flujo de trabajo principal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creation of ML models for LAI.')
    parser.add_argument('-d', '--directory', type=str, help='Output folder where the models will be stored. By default, current location', default='.')
    parser.add_argument('-m', '--model', type=str, help='Type of model', required=True, choices=BaseModelBuilder.get_model_list())
    args = parser.parse_args()

    params = eval('get_{}()'.format(args.model))
    mb = globals()['ModelBuilder' + args.model](args.model, args.directory)
    mb.build(**params)
