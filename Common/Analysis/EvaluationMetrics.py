#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EvaluationMetrics.py:
    It contains all the evaluation methods for the models.

    Source:
    https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce
    https://towardsdatascience.com/20-popular-machine-learning-metrics-part-2-ranking-statistical-metrics-22c3e5a937b6
"""
__author__ = "Jorge de la Peña García"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jpena@ucam.edu"
__status__ = "Production"

import sys
import pkg_resources
import subprocess
import json

MODULES_REQUIRED = {'numpy', 'matplotlib', 'scipy', 'scikit-plot'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = MODULES_REQUIRED - installed
if missing:
    python = sys.executable
    print([python, '-m', 'pip', 'install', *missing])
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import scikitplot as skplt
import tensorflow as tf

from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, \
    mean_squared_error, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, \
    r2_score, matthews_corrcoef, multilabel_confusion_matrix, auc
from Tools.Graphics import Graphics
from Tools.ToolsModels import is_regression_by_config, is_multiclass
from Tools.TypeML import TypeML
from Tools.Timer import Timer

class EvaluationMetrics:
    DECIMALS_ROUND = 3

    FORMAT_TEXT = '\t{:>35}:\t{:<10}'

    def __init__(self, yts, ypr, xts=None, cfg=None, model=None, id_list=None, io_data=None):
        """
        :param yts [numpy.ndarray]: test y
        :param ypr [numpy.ndarray]: prdict y
        :param xts [numpy.ndarray]: text x
        :param xts [ConfigHodler]: config class
        :param model [sklearn]:  model sklearn
        """
        EvaluationMetrics.check_input(yts, ypr, xts, model)
        self.yts = yts
        self.ypr = ypr
        self.xts = xts
        self.model = model
        self.cfg = cfg
        self.id_list = id_list
        self.io_data = io_data
        self.K = len(self.yts)
        self.plot_graphics = Graphics()

    @staticmethod
    def check_input(yts, ypr, xts, model):
        """
        Checks that the input values are correct
        """
        if not isinstance(yts, np.ndarray):
            print('Error: yts must be a numpy array')
            exit()
        if not isinstance(ypr, np.ndarray):
            print('Error: ypr must be a numpy array')
            exit()
        if not xts is None:
            if not isinstance(xts, np.ndarray):
                print('Error: xts must be a numpy array')
                exit()

    def confusion_matrix(self):
        """
                 _________________
                 |   prediction  |
        _________|_______________|
        | AC|    |   0   |   1   |
        | TU|  0 |   TN  |   FP  |
        | AL|  1 |   FN  |   TP  |
        |___|____|_______|_______|
        :return: confusion_matrix
        """
        if self.cfg:
            skplt.metrics.plot_confusion_matrix(self.yts, self.ypr)
            self.plot_graphics.save_fig(self.cfg.get_name_file_matrix_confusion())
        return multilabel_confusion_matrix(self.yts, self.ypr)

    def classification_accuracy(self):
        """
        accuracy_score = classification_accuracy
        number of correct predictions divided by the total number of predictions * 100
        :return: Classication Accuracy
        """
        return accuracy_score(self.yts, self.ypr, normalize=True)

    def precision(self):
        """
        Precision= True_Positive/ (True_Positive+ False_Positive)
        :return: precision
        """
        avg_value = 'macro' if is_multiclass(self.cfg) else 'binary'
        return precision_score(self.yts, self.ypr, average=avg_value)

    def recall(self):
        """
        Recall= True_Positive/ (True_Positive+ False_Negative)
        :return: recall
        """
        avg_value = 'macro' if is_multiclass(self.cfg) else 'binary'
        return recall_score(self.yts, self.ypr, average=avg_value)

    def f1_score(self):
        """
        F1-score= 2*Precision*Recall/(Precision+Recall)
        """
        avg_value = 'macro' if is_multiclass(self.cfg) else 'binary'
        return f1_score(self.yts, self.ypr, average=avg_value)

    def specificity(self):
        """
        Specificity = tn / (tn + fp)
        """
        if is_multiclass(self.cfg):
            spec = 0.0
            cm = multilabel_confusion_matrix(self.yts, np.rint(self.ypr)) # res = 3 matrices 2x2
            N = cm.shape[0]
            for i in range(N):
                tp = cm[i][0][0]
                fp = cm[i][0][1]
                fn = cm[i][1][0]
                tn = cm[i][1][1]
                spec = spec + (tn / (tn + tp))
            return spec / N
        else:
            tn, fp, fn, tp = confusion_matrix(self.yts, self.ypr).ravel()
            return tn / (tn + fp)

    def g_roc_curve(self):
        """
           https://github.com/dataprofessor/code/blob/master/python/ROC_curve.ipynb
        """
        if self.model and self.cfg:
            self.plot_graphics.plot_roc_curve(self.model, self.xts, self.yts, self.ypr, self.cfg.get_name_file_roc(), self.cfg)

    def save_fig(self, file):
        if self.cfg:
            plt.savefig(file, dpi=300)

    def auc_value(self):
        if is_multiclass(self.cfg):
            n_classes = len(self.id_list)
            fpr = [0] * n_classes
            tpr = [0] * n_classes
            thresholds = [0] * n_classes
            auc_score = [0] * n_classes

            for i in range(n_classes):
                fpr[i], tpr[i], thresholds[i] = roc_curve(self.yts, self.ypr, pos_label=i)
                auc_score[i] = auc(fpr[i], tpr[i])

            auc_score = np.nan_to_num(auc_score)
            return np.mean(auc_score)
        else:
            return roc_auc_score(self.yts, self.ypr)

    def mcc_value(self):
        return matthews_corrcoef(self.yts, self.ypr)

    def m_squared_error(self):
        """
        https://www.iartificial.net/error-cuadratico-medio-para-regresion/
        """
        return mean_squared_error(self.yts, self.ypr)

    def root_m_squared_error(self):
        return mean_squared_error(self.yts, self.ypr, squared=False)

    def m_absolute_error(self):
        """
        https://es.wikipedia.org/wiki/Error_absoluto_medio
        """
        return mean_absolute_error(self.yts, self.ypr)

    def rmse(self):
        return np.sqrt(self.m_squared_error())

    def pearson_correlation_coefficient(self):
        """
        https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.pearsonr.html
        """
        return scipy.stats.pearsonr(self.yts, self.ypr)

    def coefficient_of_determination(self):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        """
        return r2_score(self.yts, self.ypr)

    def print_m(self, txt, value):
        t = self.FORMAT_TEXT.format(txt, str(value)) if value else "\t{}".format(txt)
        print(t)
        if self.cfg:
            f = open(self.cfg.get_name_file_resume(), 'a')  # guardar en fichero
            f.write('{}\n'.format(t))
            f.close()

    def print_format(self, k, v):
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 == 'params':
                    self.print_m(k2, None)
                self.print_format(k2, v2)
        elif isinstance(v, list):
            for l in v:
                self.print_format(' ', str(l))
        else:
            self.print_m(k, v)

    def print_data(self):
        self.print_m("Config Data", None)
        if self.cfg:
            self.cfg.print_data()
        self.print_m("Analysis Data", None)
        for k, v in self.data.items():
            self.print_format(k, v)

    def create_json(self):
        if self.cfg:
            self.data['Config'] = self.cfg.get_config()
            with open(self.cfg.get_name_file_json(), "w") as write_file:
                json.dump(self.data, write_file, indent=4, default=lambda o: '<not serializable>')

    def all_metrics(self):
        t = Timer('Analysis')
        self.data = {}

        if not is_regression_by_config(self.cfg):
            self.data['Analysis'] = {
                'Confusion matrix': self.confusion_matrix().tolist(),
                'Accuracy': round(self.classification_accuracy() * 100, self.DECIMALS_ROUND),
                'Precision': round(self.precision() * 100, self.DECIMALS_ROUND),
                'F1': round(self.f1_score() * 100, self.DECIMALS_ROUND),
                'Recall': round(self.recall() * 100, self.DECIMALS_ROUND),
                'Specificity': round(self.specificity() * 100, self.DECIMALS_ROUND),
                'Auc': round(self.auc_value(), self.DECIMALS_ROUND),
                'MCC': round(self.mcc_value(), self.DECIMALS_ROUND)
            }
            self.g_roc_curve()
        else:
            aux = self.pearson_correlation_coefficient()
            paux = '{}/{}'.format(round(aux[0] * 100, self.DECIMALS_ROUND), round(aux[1] * 100, self.DECIMALS_ROUND))
            self.data['Analysis'] = {
                'Pearson Correlation Coefficient': paux,
                'Coefficient of Determination': round(self.coefficient_of_determination(), self.DECIMALS_ROUND),
                'Mean Absolute Error': round(self.m_absolute_error(), self.DECIMALS_ROUND),
                'Mean Squared Error': round(self.m_squared_error(), self.DECIMALS_ROUND),
                'Root Mean Squared Error': round(self.root_m_squared_error(), self.DECIMALS_ROUND)
            }
            self.plot_graphics.plot_correlation(self.yts, self.ypr, self.cfg.get_name_file_correlation())

        self.print_data()
        self.create_json()
        t.save('{}_analysis_time.txt'.format(self.cfg.get_prefix()), self.io_data)

