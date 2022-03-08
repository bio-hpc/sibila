#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel


class LearningCurveExplainer(ExplainerModel):
    def explain(self):
        return None

    def plot(self, df, method=None):
        """
            http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.plotting/#plot_learning_curves
        """
        plt.clf()
        plt.cla()
        plot_learning_curves(self.xtr, self.ytr, self.xts, self.yts, self.model)
        plt.title('Learning curves')
        plt.savefig(self.prefix + "_lc.png", dpi=300)
        plt.close()
        return None
