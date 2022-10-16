#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"


import math
import os

JOBS_SAMPLES = 1
JOBS_FEATURES = 1


""" Gets the value of an environment variable """
def env(varname, default=None):
    value = os.getenv(varname, default)
    if value.lower() in ['true', 'false']:
        return value.lower() in ['true']
    return value

""" Builds the command con call interpretability """
def interpretability_cmd():
    python_run = env("PYTHON_RUN", "python")
    cmd_exec = env("CMD_EXEC", "singularity exec")
    img_singularity = env("IMG_SINGULARITY", "Tools/Singularity/sibila.simg")

    if env("SINGULARITY", "False"):
        cmd = f"{cmd_exec} {img_singularity}"
    else:
        cmd = ''

    cmd = f"{cmd} {python_run} -m Common.Analysis.Interpretability"
    return cmd

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

