#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from Tools.IOData import IOData
"""
    DataNormalization.py:
    source: https://scikit-learn.org/stable/modules/preprocessing.html
"""
__author__ = "Jorge de la Peña García"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jpena@ucam.edu"
__status__ = "Production"


class DataNormalization:
    METHODS = {
        'MM': 'scaler_min_max',
        'MA': 'scaler_max_abs',
        'SC': 'scaler',
        'QT': 'quantile_transformer',
        'BC': 'box_cox',
        'YJ': 'yeo_johnson',
        'NE': 'normalize',
        'BI': 'binarizer',
        'PF': 'polynomialfeatures',
        'SMM': ['SC', 'MM'],
        'SMB': ['SC', 'MM', 'BI'],
        'SMA': ['SC', 'MA']
    }

    def choice_method_normalize(self, x, args):

        if args.normalize:
            for i in args.normalize:
                if isinstance(self.METHODS[i], list):
                    for m in self.METHODS[i]:
                        x = self._run_method(x, m)
                else:
                    x = self._run_method(x, i)
            return x
        else:
            return x

    def _run_method(self, x, method):
        if method in self.METHODS.keys():
            func = getattr(self, self.METHODS[method])
            return func(x)
        else:
            IOData.print_e("Error normalized")
            exit()

    @staticmethod
    def scaler(x):
        """
        Scaled data has zero mean and unit variance:
        @param x: numpy.ndarray:
        @return:
        """
        return preprocessing.scale(x)

    @staticmethod
    def scaler_min_max(x, min=0, max=1):
        """
            Scale data between min and max
            @param x: numpy.ndarray:
        """
        min_max_scaler = preprocessing.MinMaxScaler([min, max])
        return min_max_scaler.fit_transform(x)

    @staticmethod
    def scaler_max_abs(x):
        """
            dividing through the largest maximum value in each feature. It is meant f
            or data that is already centered at zero or sparse data.
            @param x: numpy.ndarray:
        """
        max_abs_scaler = preprocessing.MaxAbsScaler()
        return max_abs_scaler.fit_transform(x)

    @staticmethod
    def quantile_transformer(x, random_state=2020):
        """
            QuantileTransformer and quantile_transform provide a non-parametric t
            ransformation to map the data to a uniform distribution with values between 0 and 1:
            @param x: numpy.ndarray:
        """
        quantile_transformer = preprocessing.QuantileTransformer(random_state=random_state)
        x_aux = quantile_transformer.fit_transform(x)
        return (x_aux)

    @staticmethod
    def box_cox(x):
        """
            Box-Cox can only be applied to strictly positive data. In both methods, the transformation is
            parameterized by ,which is determined through maximum likelihood estimation. Here is an example of
            using Box-Cox to map samples drawn from a lognormal distribution to a normal distribution:
            @param x: numpy.ndarray:
        """
        try:
            return preprocessing.PowerTransformer(method='box-cox', standardize=False).fit_transform(x)
        except (ValueError, KeyError):
            IOData.print_e("Error normalized,  it could be that the matrix is not strictly positive")

    @staticmethod
    def yeo_johnson(x):
        """
            The Yeo-Johnson transformation is very similar to the Box-Cox but does not require the input
            variables to be strictly positive.
            @param x: numpy.ndarray:
        """
        return preprocessing.PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(x)

    @staticmethod
    def normalize(x, norm='l2'):
        """
        Normalize samples individually to unit norm.
        @param x: numpy.ndarray:
        @param norm: The norm to use to normalize each non zero sample.
        """
        return preprocessing.Normalizer(norm=norm).fit_transform(x)

    @staticmethod
    def binarizer(x, threshold=0.5):
        """
        Feature binarization is the process of thresholding numerical features to get boolean values
        @param x: dataset
        @param threshold: cutoff
        @return:
        """
        return preprocessing.Binarizer(threshold=threshold).fit_transform(x)

    @staticmethod
    def polynomialfeatures(x, degree=2, interaction_only=False):
        """
            Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.
        @param x:
        @param degree: integer The degree of the polynomial features.
        @param interaction_only: only interaction features are produced
        @return:
        """
        return preprocessing.PolynomialFeatures(degree=degree, interaction_only=interaction_only).fit_transform(x)
