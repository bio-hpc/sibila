"""
    consensus_by_xai_approach.py:

    Makes a consensus of the interpretability on different ML models
"""

import argparse
from glob import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from os.path import basename, splitext, isfile, join, isdir
from pathlib import Path

DEBUG = False
MAX_FEATURES = 10
DEFAULT_CUTOFF = 0.75
FOLDER_OUT = "Consensus/"
INTERPRETABILITY = {
    'LIME': '/LIME/*.csv',
    'Shapley': '/Shapley/csv/*.csv',
    'Integrated Gradients': '/Integrated_Gradients/csv/*.csv',
    'DiCE': '/DICE/csv/*.csv',
    'Permutation Importance': '/*PermutationImportance.csv',
    'RF Permutation Importance': '/*RFPermutationImportance.csv',
    'Anchor': '/*Anchor.csv'
}
METRICS = {
    "AUC": "Auc",
    "F1": "F1",
    "ACC": "Accuracy",
    "PRE": "Precision",
    "REC": "Recall",
    "SPE": "Specificity",
    "PCC": "Pearson Correlation Coefficient",
    "R2": "Coefficient of Determination",
    "MAE": "Mean Absolute Error",
    "MSE": "Mean Squared Error"
}
FEATURES = []
N_FEATURES = 0
MODEL_METRICS = {}

def print_f(msg):
    if DEBUG:
        print(msg)

def create_dir(dir):
    if not isdir(dir):
        os.mkdir(dir)

def find_models(folder, metric, cutoff):
    global MODEL_METRICS

    models = []
    lst_data = glob(folder + '/*_data.json')
    for foo in lst_data:
        with open(foo, 'r') as f:
            data = json.load(f)
            score = float(data['Analysis'][METRICS[metric]])
            if score >= cutoff:
                models.append(data['Config']['Model_params']['model'])
                MODEL_METRICS[data['Config']['Model_params']['model']] = score

    return models

def search_files(exp, folder, models):
    pattern = INTERPRETABILITY[exp]
    aux = glob(folder + pattern)
    lst_csv = []

    for f in aux:
        model = basename(f).split('_')[0]
        if model in models:
            lst_csv.append(f)

    return lst_csv

def process_data(lst_csv):
    global N_FEATURES, FEATURES, MODEL_METRICS

    data = []
    for foo in lst_csv:
        model = basename(foo).split('_')[0]
        feat, attr, rank = [], [], []

        f = open(foo, 'r')
        lines = f.readlines()[1:]
        N_FEATURES = max(N_FEATURES, len(lines))

        for i, l in enumerate(lines):
            aux = l.split(',')
            if not 'Sum' in aux[0]:
                feat.append(aux[0])
                attr.append(float(aux[1]) * MODEL_METRICS[model])
                rank.append(min(i+1, MAX_FEATURES+1))
        f.close()

        data.append(pd.DataFrame({'feature': feat, 'attribution': attr, 'rank': rank}))

        if len(lines) == N_FEATURES:
            FEATURES = feat.copy()

    return data

def add_missing_features(df, features):
    df_feats = df['feature'].to_numpy()
    for f in features:
        if f not in df_feats:
            df = df.append({'feature': f, 'attribution': 0.0, 'rank': MAX_FEATURES+1}, ignore_index=True)
    return df    

def plot_global(df, title, out_file, x_title='Average Score'):
    labels = df.iloc[:,0].to_numpy()
    attrs = df.iloc[:,1].to_numpy()
    errors = df.iloc[:,2].to_numpy()
    
    barlist = plt.bar(labels, attrs, yerr=errors, error_kw=dict(lw=0.75, capsize=1), align='center', alpha=0.5)
    plt.xticks(fontsize=8, rotation=45)
    plt.title(title)
    plt.xlabel(x_title)
    plt.tight_layout()

    for i in range(len(labels)):
        if 'Sum' in labels[i]:
            barlist[i].set_color('r')

    plt.savefig(out_file)
    plt.close()

def summarize(df, key):
    df2 = df.copy()

    if len(df2) > MAX_FEATURES:
        n_others = len(df2) - MAX_FEATURES
        title = 'Sum other {} features'.format(str(n_others))
        df_others = pd.DataFrame(data=[[title, df2[MAX_FEATURES:][key].mean()]], columns=['feature', key])
        df2 = pd.concat([df2[:MAX_FEATURES], df_others], ignore_index=True)

    return df2

def persist_data(df, title, filename, x_title=None, ascending=True):
    df = df.reindex(df['attribution'].abs().sort_values(ascending=ascending).index)
    df.to_csv('{}.csv'.format(filename), index=False)
    df_mean = summarize(df, 'attribution')
    plot_global(df_mean, title, '{}.png'.format(filename), x_title=x_title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Folder where the files are', required=True)
    parser.add_argument('-c', '--cutoff', help='Cut off', default=DEFAULT_CUTOFF, type=float)
    parser.add_argument('-m', '--metric', help='Metric to use', default='AUC', choices=METRICS.keys())
    args = parser.parse_args()

    dir_out = join(args.folder, FOLDER_OUT)
    create_dir(dir_out)

    dfs = []
    models = {}
    for exp in INTERPRETABILITY:
        models[exp] = find_models(args.folder, args.metric, args.cutoff)
        if len(models[exp]) == 0:
            raise Exception('No model was found with {} >= {}'.format(args.metric, args.cutoff))

        lst_csv = search_files(exp, args.folder, models[exp])
        if len(lst_csv) == 0:
            print_f('No files found for {} in {}'.format(exp, args.folder))
            continue
        print_f(lst_csv)

        df = process_data(lst_csv)
        dfs.append(df)

    for i, exp in enumerate(INTERPRETABILITY):
        # complete missing features
        for j in range(len(dfs[i])):
            if len(dfs[i][j]) < N_FEATURES:
                dfs[i][j] = add_missing_features(dfs[i][j], FEATURES)
        
        # join and aggregate all the data
        df = pd.concat(dfs[i], ignore_index=True)
        print_f(df)       

        df_agg = df.groupby('feature').agg(['mean','std']).reset_index()
        df_agg.columns = ['feature','attribution_mean','attribution_std', 'rank_mean', 'rank_std']
        
        # as the previous calculation considered the sign, do it over with the abs
        df_agg['attribution_mean'] = df.groupby('feature').attribution.apply(lambda c: c.abs().mean()).reset_index()['attribution']
        df_agg['attribution_std'] = df.groupby('feature').attribution.apply(lambda c: c.abs().std()).reset_index()['attribution']
        df_agg = df_agg.fillna(0.0)
        
        # segregate the data between attribution and ranking
        df_attr = pd.DataFrame({'feature': df_agg['feature'], 'attribution': df_agg['attribution_mean'], 'std': df_agg['attribution_std']})
        df_rank = pd.DataFrame({'feature': df_agg['feature'], 'attribution': df_agg['rank_mean'], 'std': df_agg['rank_std']})

        print_f(df_attr)
        print_f(df_rank)

        title = '{} - {} ({}>={})'.format(exp, ' '.join(models[exp]), args.metric, args.cutoff)
        filename = '{}/{}{}_strategy_attribution'.format(dir_out, exp, '_'.join(models[exp]))
        persist_data(df_attr, title, filename, x_title='Average Attribution', ascending=False)

        filename = '{}/{}{}_strategy_rank'.format(dir_out, exp, '_'.join(models[exp]))
        persist_data(df_rank, title, filename, x_title='Average Rank')
