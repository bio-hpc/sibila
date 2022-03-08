#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TypeML.py
    Type of machine learning task: classification or regression
"""
__author__ = "Antonio Jesús Banegas Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

from enum import Enum

class TypeML(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

