#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JobManager.py:

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import math
import os
from Tools.PostProcessing.Serialize import Serialize
from Tools.IOData import serialize_class
from simple_slurm import Slurm


class JobManager:

    BATCH_SIZE = 3 # 100

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

    def send_jobs(self, params, methods, foo, n_jobs):
        cfg = params['cfg']
        name_model = cfg.get_params()['model']
        job_folder = params['io_data'].get_job_folder()
        job_list = ''
        
        # send an independent job for every interpretability method
        for method in methods:
            slurm = Slurm(
                array = range(0, n_jobs),
                cpus_per_task = 2,
                job_name = 'SIBILA-{}-{}'.format(method, name_model),
                output = '{}{}-{}-{}.out'.format(job_folder, method, name_model, Slurm.JOB_ARRAY_ID),
                error = '{}{}-{}-{}.err'.format(job_folder, method, name_model, Slurm.JOB_ARRAY_ID),
                time = '72:00:00'
            )
            job_id = slurm.sbatch('python3 -m Common.Analysis.Interpretability {} {} {}'.format(foo, method, str(Slurm.SLURM_ARRAY_TASK_ID)))
            job_list += '{},'.format(job_id)

        # send a final job to finish the process once all the previous ones are done
        slurm2 = Slurm(
            cpus_per_task = 1,
            job_name = 'SIBILA-END',
            output = '{}end.out'.format(job_folder),
            error = '{}end.err'.format(job_folder),
            time = '4:00:00',
            dependency = dict(afterany = job_list[:-1])
        )
        slurm2.sbatch('python3 -m Common.Analysis.EndProcess {}/{}'.format(os.getcwd(), cfg.get_folder()))

