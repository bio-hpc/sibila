#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

#import pandas as pd
#from Tools.ToolsModels import is_regression_by_config
#from Tools.Graphics import Graphics
#import numpy as np
#import re
#from Tools.HTML.LIMEHTMLBuilder import LIMEHTMLBuilder
#from lime import lime_tabular
#from tqdm import tqdm
#from pathlib import Path
from ConsensusBase import ConsensusBase
#from Tools.ToolsModels import is_rulefit_model
#from Common.Config.ConfigHolder import ATTR, COLNAMES, FEATURE, STD, PROBA

class ConsensusAverageMean(ConsensusBase):

    def __init__(self, folder):
        super(ConsensusAverageMean, self).__init__(folder)

    def consensus(self):
        print("ejecutando media aritmetica")
        pass

