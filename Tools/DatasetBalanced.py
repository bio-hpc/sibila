#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from Tools.IOData import IOData
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from Tools.ToolsModels import is_penalty_weighted, is_regression_by_config


def print_size(x, y):
    print("_________________-")
    print("1 " + str(np.count_nonzero(y == 1)))
    print("0 " + str(np.count_nonzero(y == 0)))
    print(x.shape)
    print(y.shape)


class DatasetBalanced:
    METHODS = {'ADASYN': 'adasyn', 'ROS': 'random_over_sample', 'SMOTE': 'smote', 'PEN': 'weight_penalty', 'RUS': 'random_under_sample'}

    def choice_method_balanced(self, x, y, args, idx_samples):

        if args.balanced:
            for i in args.balanced:
                if isinstance(self.METHODS[i], list):
                    for m in self.METHODS[i]:
                        x, y = self._run_method(x, y, m, random_state=args.seed)
                else:
                    x, y = self._run_method(x, y, i, random_state=args.seed)
            if (len(y) != len(idx_samples)):
                diff_lst = len(y) - len(idx_samples)
                for i in range(diff_lst):
                    idx_samples.append('{}'.format(i * -1))

            return x, y, idx_samples
        else:
            return x, y, idx_samples

    def _run_method(self, x, y, method, random_state=None):
        if method in self.METHODS.keys():
            func = getattr(self, self.METHODS[method])
            return func(x, y, random_state=random_state)
        else:
            IOData.print_e("Error normalized")
            exit()

    @staticmethod
    def random_over_sample(x, y, random_state=None):
        """

        @param x: numpy.ndarray:
        @param y: numpy.ndarray:

        @return:
        """
        print_size(x, y)
        os = RandomOverSampler(sampling_strategy='auto')
        x, y = os.fit_resample(x, y)
        print_size(x, y)
        return x, y

    @staticmethod
    def random_under_sample(x, y, random_state=None):
        """

        @param x: numpy.ndarray:
        @param y: numpy.ndarray:

        @return:
        """
        print_size(x, y)
        os = RandomUnderSampler(sampling_strategy='auto')
        x, y = os.fit_resample(x, y)
        print_size(x, y)
        return x, y

    @staticmethod
    def adasyn(x, y, random_state=None):
        """

        @param x: numpy.ndarray:
        @param y: numpy.ndarray:

        @return:
        """
        print_size(x, y)

        n_neighbors = 5
        while True:
            try:
                os = ADASYN(random_state=random_state, n_neighbors=n_neighbors, n_jobs=-1)
                x, y = os.fit_resample(x, y)
            except:
                n_neighbors -= 1

        print_size(x, y)
        return x, y

    @staticmethod
    def smote(x, y, random_state=None):
        """

        @param x: numpy.ndarray:
        @param y: numpy.ndarray:

        @return:
        """
        print_size(x, y)

        k_neighbors = 5
        while True:
            try:
                os = SMOTE(random_state=random_state, k_neighbors=k_neighbors, n_jobs=-1)
                x, y = os.fit_resample(x, y)
                break
            except:
                k_neighbors -= 1

        print("SMOTE")
        print_size(x, y)
        return x, y

    @staticmethod
    def weight_penalty(x, y, random_state=None):
        """
        @param x: numpy.ndarray:
        @param y: numpy.ndarray:

        @return:
        """
        return x, y

    @staticmethod
    def get_class_weights(model, y, cfg):
        """
        @param model: model:
        @param y: numpy.ndarray:
        @param cfg: Common.Config.ConfigHolder:
        """
        if is_regression_by_config(cfg):
            return None

        ytr_weights = [1.0] * len(np.unique(y))
        if is_penalty_weighted(cfg.get_args()) and any(
                s in str(model) for s in ['SVC', 'DecisionTreeClassifier', 'RandomForestClassifier', 'tensorflow']):
            ytr_weights = len(y) / (len(np.unique(y)) * np.bincount(y))

        class_weights = dict(zip(np.arange(len(ytr_weights)).astype(int).tolist(), ytr_weights))
        cfg.get_config()['Penalty_weights'] = str(class_weights)
        return class_weights
