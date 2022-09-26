#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JobManager.py:

Splits data into blocks of fixed size and launch a job for computing the interpretability
of each block. Afterwards, another job is submitted to collect all the results.
"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import math
import os
import subprocess
from Tools.PostProcessing.Serialize import Serialize
from Tools.IOData import serialize_class
from .jobs import interpretability_cmd, build_blocks

class JobManager:

    SCRIPT_PATH = 'Tools/Bash/Queue_manager/SLURM.sh'
    INTERP_PATH = interpretability_cmd() #('singularity exec Tools/Singularity/sibila.simg ' if env("SINGULARITY") else '') + 'python3 -m Common.Analysis.Interpretability'

    def parallelize(self, params, methods):
        # serialize params for upcoming jobs
        serialized_params = self.serialize_params(params)

        # split test data into regular blocks
        self.send_jobs(params, methods, serialized_params)

    def serialize_params(self, params):
        foo = params['cfg'].get_prefix() + '_params.pkl'
        class_serializer = Serialize(**params)
        if not os.path.isfile(foo):
            serialize_class(class_serializer, params['cfg'].get_prefix() + '_params.pkl')
        return foo

    def send_jobs(self, params, methods, foo):
        cfg = params['cfg']
        name_model = cfg.get_params()['model']
        job_folder = params['io_data'].get_job_folder()

        # send an independent job for every interpretability method
        for method in methods:
             block_ids = build_blocks(method)

             for index in block_ids:
                 base_name = f"{method}-{name_model}-{index}"
                 name_script = f"{job_folder}/{base_name}.sh"
                 name_job = f"{base_name}-SIBILA"
                 os.system('sh {}/{} {} {} {} {} > {}'.format(os.getcwd(), JobManager.SCRIPT_PATH, job_folder, name_job, '4:00:00', '2', name_script)) # TODO configure this externally
                 os.system(f'echo "{JobManager.INTERP_PATH} {foo} {method} {index}" >> {name_script}')

