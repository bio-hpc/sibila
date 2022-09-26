#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"


import math
import os

JOBS_SAMPLES = 10
JOBS_FEATURES = 2

""" Gets the value of an environment variable """
def env(varname):
    return os.getenv(varname, 'False').lower() in ('true', '1', 't')

""" Check if an interpretability method is related to samples or features """
def is_feature_method(method):
    if method in ['PDP', 'ALE']:
        return True
    return False

""" Block IDs are calculated depending on the method """
def build_blocks(method):
    if is_feature_method(method):
        return [i for i in range(JOBS_FEATURES)]
    
    return [i for i in range(JOBS_SAMPLES)]

""" Returns the number of items in each block """
def get_nitems_per_block(method, shape):
    if is_feature_method(method):
        return math.ceil(shape[1] / JOBS_FEATURES)

    return math.ceil(shape[0] / JOBS_SAMPLES)

