#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Common.Analysis.Interpretability import Interpretability
from Tools.datasets import get_dataset, split_samples, FIELD_TARGET
from Tools.IOData import IOData
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
from os.path import join, basename, splitext
from Tools.PostProcessing.Serialize import Serialize
from Tools.DatasetBalanced import DatasetBalanced
import numpy as np
import pandas as pd
from Models import *
from Tools.Timer import Timer
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def get_cfg(folder_experiment, file_dataset, type_model, args):
    params = get_config(type_model, args)
    return ConfigHolder(file_dataset, folder_experiment, args, params)


def get_basic_cfg(folder_experiment, file_dataset, args):
    params = get_basic_config(args)
    return ConfigHolder(file_dataset, folder_experiment, args, params)


def main():

    args = InputParams().read_params()
    file_dataset = args.dataset
    options = args.option

    io_data = IOData()

    if not args.introduced_folder:

        args.folder = '{}_{}'.format(args.folder, datetime.now().strftime("%Y-%m-%d"))
        io_data.create_dirs(args.folder)
    else:
        io_data.create_dirs_no_remove(args.folder)

    t = Timer('Load data')
    x, y, id_list, idx_samples = get_dataset(file_dataset, io_data)

    x = DataNormalization().choice_method_normalize(x, args)

    Graphics().graph_dataset(x, y, id_list, FIELD_TARGET, join(args.folder, 'Dataset/'))
    t.save('{}/load_time.txt'.format(args.folder), io_data)

    if args.model:
        if is_regression_by_config(get_basic_cfg("", file_dataset, args)):
            y = DataNormalization().choice_method_normalize(y.reshape(-1, 1), args).ravel()
        x, y, idx_samples = DatasetBalanced().choice_method_balanced(x, y, args, idx_samples)
        [execute_pred(x, y, id_list, idx_samples, io_data, args.folder, file_dataset, type_model, args) for type_model in args.model]
    else:
        # you look at the first option to know if it is regression. If it is regression, then y is also normalized
        if is_regression_by_config(get_cfg("", file_dataset, options[0], args)):
            y = DataNormalization().choice_method_normalize(y.reshape(-1, 1), args).ravel()
        x, y, idx_samples = DatasetBalanced().choice_method_balanced(x, y, args, idx_samples)

        [
            execute(x, y, id_list, idx_samples, io_data, args.folder, file_dataset, type_model, args)
            for type_model in options
        ]
        MergeResults(args.folder)

    if not args.queue:
        EndProcess(args.folder)


def execute(x, y, id_list, idx_samples, io_data, folder_experiment, file_dataset, type_model, args):
    cfg = get_cfg(folder_experiment, file_dataset, type_model, args)

    model = globals()[type_model](io_data, cfg, id_list)
    print("\n")
    cfg.set_prefix(model.get_prefix())
    xtr, xts, ytr, yts, idx_xtr, idx_xts = split_samples(x, y, (args.trainsize / 100), io_data, args.seed, idx_samples)

    t = Timer('Training')
    model.train(xtr, ytr)
    t.save('{}_training_time.txt'.format(cfg.get_prefix()), io_data)

    ypr = model.predict(xts)
    BaseModel.save_model(cfg, model.get_model())
    EvaluationMetrics(yts, ypr, xts, cfg, model.get_model(), id_list, io_data).all_metrics()
    Interpretability(Serialize(model.get_model(), xtr, ytr, xts, yts, id_list, cfg, io_data, idx_xts))


def execute_pred(x, y, id_list, idx_samples, io_data, folder_experiment, file_dataset, type_model, args):
    cfg = get_basic_cfg(folder_experiment, file_dataset, args)
    model = BaseModel.load(type_model)
    print("\n")
    cfg.set_prefix(join(args.folder, basename(type_model)))

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

    ypr_class = np.squeeze(ypr_class)
    ypr_prob = np.squeeze(ypr_prob)

    cfg.set_time_end()

    # export predictions to csv
    outfile = str(int(datetime.now().timestamp()))
    df = pd.DataFrame({'Sample ID': idx_samples, 'Predicted class': ypr_class, 'Probability': ypr_prob})
    df.to_csv('prediction_{}.csv'.format(outfile), index=False)
    print('Results saved in prediction_{}.csv'.format(outfile))

    exit()

if __name__ == "__main__":
    main()
