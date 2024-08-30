#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Common.Analysis.Interpretability import Interpretability
from Tools.datasets import get_dataset, split_samples, FIELD_TARGET
from Tools.IOData import IOData, get_serialized_params
from Common.Config.ConfigHolder import ConfigHolder
from Common.Config.config import get_config, get_basic_config
from Common.Analysis.EvaluationMetrics import EvaluationMetrics, TypeML
from Common.Analysis.EndProcess import EndProcess
from Common.Analysis.MergeResults import MergeResults
from Tools.Graphics import Graphics
from Common.Input.InputParams import InputParams
import datetime
from datetime import datetime
from Tools.DataNormalization import DataNormalization
from Tools.ToolsModels import is_regression_by_config, is_tf_model
import os
from os.path import join, basename, splitext, dirname, exists
from Tools.Serialize import Serialize
from Tools.DatasetBalanced import DatasetBalanced
import numpy as np
import pandas as pd
from Models import *
from Tools.Timer import Timer
from Tools.Bash.Queue_manager.JobManager import JobManager
from Tools.GPUTracker import GPUTracker
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def get_cfg(folder_experiment, file_dataset, type_model, args):
    params = get_config(type_model, args)
    return ConfigHolder(file_dataset, folder_experiment, args, params)

def get_basic_cfg(folder_experiment, file_dataset, args):
    params = get_basic_config(args)
    return ConfigHolder(file_dataset, folder_experiment, args, params)

def save_params(serialize):
    jm = JobManager()
    return jm.serialize_params(serialize.get_params())

def main():

    args = InputParams().read_params()
    file_dataset = args.dataset
    options = args.option

    io_data = IOData()

    if args.explanation is not None:
        Interpretability(get_serialized_params(args.explanation))
        exit()

    if args.model is not None:
        args.folder = dirname(args.model[0])
        io_data.create_dirs_no_remove(args.folder)
    elif not args.introduced_folder:
        args.folder = '{}_{}'.format(args.folder, datetime.now().strftime("%Y-%m-%d"))
        io_data.create_dirs(args.folder)
    else:
        io_data.create_dirs_no_remove(args.folder) 

    t = Timer('Load data')
    x, y, id_list, idx_samples, n_classes = get_dataset(file_dataset, io_data, args.model)
    x = DataNormalization().choice_method_normalize(x, args)

    if not args.model and not args.skip_dataset_analysis:
        Graphics().graph_dataset(x, y, id_list, FIELD_TARGET, join(args.folder, 'Dataset/'))
    t.save('{}/load_time.txt'.format(args.folder), io_data)

    if args.model:
        if is_regression_by_config(get_basic_cfg("", file_dataset, args)):
            y = DataNormalization().choice_method_normalize(y.reshape(-1, 1), args).ravel()
        #x, y, idx_samples = DatasetBalanced().choice_method_balanced(x, y, args, idx_samples)
        [execute_pred(x, y, id_list, idx_samples, io_data, args.folder, file_dataset, type_model, args) for type_model in args.model]
    else:
        # you look at the first option to know if it is regression. If it is regression, then y is also normalized
        if is_regression_by_config(get_cfg("", file_dataset, options[0], args)):
            y = DataNormalization().choice_method_normalize(y.reshape(-1, 1), args).ravel()
        #x, y, idx_samples = DatasetBalanced().choice_method_balanced(x, y, args, idx_samples)
        [
            execute(x, y, id_list, idx_samples, io_data, args.folder, file_dataset, type_model, args, n_classes)
            for type_model in options
        ]

    if not args.queue:
        MergeResults(args.folder)
        EndProcess(args.folder)

def execute(x, y, id_list, idx_samples, io_data, folder_experiment, file_dataset, type_model, args, n_classes):
    cfg = get_cfg(folder_experiment, file_dataset, type_model, args)
    is_regression = is_regression_by_config(cfg)

    model = globals()[type_model](io_data, cfg, id_list)
    print("\n")
    cfg.set_prefix(model.get_prefix())

    gt = GPUTracker(cfg.get_prefix())
    gt.start(type_model)

    xtr, xts, ytr, yts, idx_xtr, idx_xts = split_samples(x, y, (args.trainsize / 100), io_data, args.seed, idx_samples, is_regression=is_regression)
    xtr, ytr, idx_samples = DatasetBalanced().choice_method_balanced(xtr, ytr, args, idx_samples)

    t = Timer('Training')
    model.train(xtr, ytr)
    t.save('{}_training_time.txt'.format(cfg.get_prefix()), io_data)

    ypr = model.predict(xts)
    BaseModel.save_model(cfg, model.get_model())
    EvaluationMetrics(yts, ypr, xts, cfg, model.get_model(), id_list, io_data, n_classes).all_metrics()

    sp = Serialize(model.get_model(), xtr, ytr, xts, yts, id_list, cfg, io_data, idx_xts)
    pkl_file = save_params(sp)
    io_data.print_m("Model's state saved in {}".format(pkl_file))

    gt.stop()
    gt.plot()

    if not args.skip_interpretability:
        Interpretability(sp)


def execute_pred(x, y, id_list, idx_samples, io_data, folder_experiment, file_dataset, type_model, args):
    cfg = get_basic_cfg(folder_experiment, file_dataset, args)
    model = BaseModel.load(type_model)
    print("\n")
    cfg.set_prefix(join(args.folder, basename(type_model)))

    gt = GPUTracker(cfg.get_prefix())
    gt.start(type_model)

    ypr_class, ypr_prob = [], []
    for xts in x:
        try:
            if 'predict_proba' in dir(model):
                yhat = model.predict_proba(np.array([xts]))
            else:
                yhat = model.predict(np.array([xts]))
        except:
            xts = tf.expand_dims(xts, -1)
            yhat = model.predict(np.array([xts]))
    
        if not args.regression:
            ypr_class.append(yhat.argmax(axis=1))
            ypr_prob.append(np.amax(yhat, axis=1))
        else:
            ypr_class.append(-1)
            ypr_prob.append(np.amax(yhat))

    ypr_class = np.squeeze(ypr_class)
    ypr_prob = np.squeeze(ypr_prob)

    cfg.set_time_end()

    # export predictions to csv
    outfile = 'prediction_{}__{}.csv'.format(splitext(basename(type_model))[0], splitext(basename(args.dataset))[0])
    df = pd.DataFrame({'Sample ID': idx_samples, 'Predicted class': ypr_class, 'Probability': ypr_prob})
    df.to_csv(outfile, index=False)
    print('Results saved in {}'.format(outfile))

    gt.stop()
    gt.plot()

    exit()

if __name__ == "__main__":
    main()
