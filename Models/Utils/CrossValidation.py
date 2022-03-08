#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import model_selection
from Tools.IOData import IOData
import numpy as np
"""
    CrossValidation.py:
    sources: 
    	https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_469
    	https://www.cienciadedatos.net/documentos/30_cross-validation_oneleaveout_bootstrap
    	https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    'n_splits' and 'random_state' are always required for compatibility. 
    They can be omitted when have no meaning and will set to None.
"""
__author__ = "Antonio Jesús Banegas Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"


class CrossValidation:
    METHODS = {
        'GKF': 'group_kfold',
        'KF': 'kfold',
        'LOO': 'loo',
    	'RKF': 'repated_kfold',
        'SS': 'shuffle_split',
        'ST': 'stratified_kfold'
    }

    N_SPLITS = 5
    RANDOM_STATE = 500

    def __init__(self, io_data):
        self.io_data = io_data

    def choice_method(self, method):
        #if isinstance(self.METHODS[method], list):
        #    for m in self.METHODS[method]:
        #        return getattr(self, self.METHODS[m])
        #else:
        if method in self.METHODS.keys():
            return getattr(self, self.METHODS[method])
        else:
            IOData.print_e('Invalid cross-validation method')
            exit()

    def run_method(self, method, x, y, custom_fn, **kwargs):
        if method:
            cv_idx = 1
            for train_idx, test_idx in method(x, y, **kwargs):
                self.io_data.print_m('\n\tRunning Cross-Validation {}'.format(cv_idx))
                xtr, ytr = np.take(x, train_idx, axis=0), y[train_idx]
                xts, yts = np.take(x, test_idx, axis=0), y[test_idx]

                # call user function to customize every cross validation
                custom_fn(xtr, ytr)
                
                cv_idx += 1            
        else:
            IOData.print_e('Invalid method object')
            exit()

    @staticmethod
    def kfold(x, y, n_splits=5, shuffle=False, random_state=RANDOM_STATE):
        """
        K-fold cross-validation. 
        Split dataset into k consecutive folds. Each fold is then used once as a validation 
        while the k - 1 remaining folds form the training set.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold

        @param x: numpy.ndarray
        @param y: numpy.ndarray
        @n_splits: int
        @shuffle: bool
        @random_state: int
        """
        if not shuffle:
            random_state = None
        kf = model_selection.KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        return kf.split(x, y)

    @staticmethod
    def repated_kfold(x, y, n_splits=5, n_repeats=10, random_state=RANDOM_STATE):
        """
        Repeated K-fold cross-validation. 
        Repeats K-Fold n times with different randomization in each repetition.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold

        @param x: numpy.ndarray
        @param y: numpy.ndarray
        @n_splits: int
        @n_repeats: int
        @random_state: int
        """
        rskf = model_selection.RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return rskf.split(x, y)

    @staticmethod
    def loo(x, y, n_splits=None, random_state=None):
        """
        Leave-one-out cross-validation. 
        Each sample is used once as a test set (singleton) while the remaining samples form the training set.
        LeaveOneOut() is equivalent to KFold(n_splits=n) and LeavePOut(p=1) where n is the number of samples.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut

        @param x: numpy.ndarray
        @param y: numpy.ndarray
        """
        loo = model_selection.LeaveOneOut()
        return loo.split(x, y)

    @staticmethod
    def shuffle_split(x, y, n_splits=10, test_size=None, train_size=None, random_state=RANDOM_STATE):
        """
        Random permutation cross-validator.
        Yields indices to split data into training and test sets.
        Note: contrary to other cross-validation strategies, random splits do not guarantee 
        that all folds will be different, although this is still very likely for sizeable datasets.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

        @param x: numpy.ndarray
        @param y: numpy.ndarray
        @n_splits: int
        @test_size: float
        @train_size: float
        @random_state: int
        """
        ss = model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
        return ss.split(x, y)

    @staticmethod
    def stratified_kfold(x, y, n_splits=5, shuffle=True, random_state=RANDOM_STATE):
        """
        Stratified K-folds cross-validator.
        This cross-validation object is a variation of KFold that returns stratified folds.
        The folds are made by preserving the percentage of samples for each class.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

        @param x: numpy.ndarray
        @param y: numpy.ndarray
        @n_splits: int
        @shuffle: bool
        @random_state: int
        """
        skf = model_selection.StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        return skf.split(x, y)

    @staticmethod
    def group_kfold(x, y, n_splits=5, n_groups=5, random_state=None):
        """
        K-fold iterator variant with non-overlapping groups.
        The same group will not appear in two different folds (the number of distinct groups has to be at least equal to the number of folds).
        The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold

        @param x: numpy.ndarray
        @param y: numpy.ndarray
        @n_splits: int
        @n_groups: int
        """        
        groups = np.floor(np.linspace(0, n_groups, len(y)))
        gkf = model_selection.GroupKFold(n_splits=n_splits)
        return gkf.split(x, y, groups)
