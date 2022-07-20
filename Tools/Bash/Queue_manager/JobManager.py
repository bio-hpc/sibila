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


class JobManager:

    BATCH_SIZE = 3
    JOBS = []
    SCRIPT_PATH = 'Tools/Bash/Queue_manager/SLURM.sh'
    INTERP_PATH = 'singularity exec Tools/Singularity/sibila.simg python3 -m Common.Analysis.Interpretability' # TODO check if singularity is being used

    def parallelize(self, params, methods):
        # serialize params for upcoming jobs
        serialized_params = self.serialize_params(params)

        # split test data into regular blocks
        N = len(params['xts'])
        n_jobs = math.ceil(N/self.BATCH_SIZE)
        self.send_jobs(params, methods, serialized_params, n_jobs)

    def serialize_params(self, params):
        foo = params['cfg'].get_prefix() + '_params.pkl'
        class_serializer = Serialize(**params)
        if not os.path.isfile(foo):
            serialize_class(class_serializer, params['cfg'].get_prefix() + '_params.pkl')
        return foo

    def interpretation_job(self, method, name_model, index, folder, foo):
        # create the script
        base_name = f"{method}-{name_model}-{index}"
        name_script = f"{folder}/{base_name}.sh"
        name_job = f"{base_name}-SIBILA"
        os.system('sh {}/{} {} {} {} {} > {}'.format(os.getcwd(), JobManager.SCRIPT_PATH, folder, name_job, '4:00:00', '1', name_script))
        os.system(f'echo "{JobManager.INTERP_PATH} {foo} {method} {index}" >> {name_script}')

        # run the script on the queue
        out = subprocess.check_output(f"sbatch {name_script}", shell=True)
        job_id = (str(out).split(' ')[-1])[:-3]
        return job_id

    def send_jobs(self, params, methods, foo, n_jobs):
        cfg = params['cfg']
        name_model = cfg.get_params()['model']
        job_folder = params['io_data'].get_job_folder()
        block_ids = [i for i in range(n_jobs)]

        # send an independent job for every interpretability method
        for method in methods:
             for index in block_ids:
                 job_id = self.interpretation_job(method, name_model, index, job_folder, foo)
                 JobManager.JOBS.append(job_id)

    @staticmethod
    def end_process(folder):
        directory = os.getcwd() + '/' + folder
        os.system("sbatch --dependency=afterany:{} {}/end_job.sh".format(','.join(JobManager.JOBS), directory))

