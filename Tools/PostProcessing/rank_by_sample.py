"""
    Consenso_premutation.py:

    Makes a consensus of the interpretability (RF, DT, SVM...) of the different ML models (Lime, Permutation...)

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import argparse
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import basename, splitext, isfile, join, isdir
from pathlib import Path

DEBUG = True
DELIMITER = '##'
MAX_FEATURES = 10
FOLDER_OUT = "Consensus/"
MODELS = ['ANN', 'DT', 'KNN', 'RF', 'RLF', 'RP', 'SVM', 'XGBOOST']
INTERPRETABILITY = {
    'LIME': '/LIME/',
    'Shapley': '/Shapley/csv/',
    'Integrated Gradients': '/Integrated_Gradients/csv/',
    'DiCE': '/DICE/csv/'
}

def create_dir(dir):
    if not isdir(dir):
        os.mkdir(dir)

def search_files(model, folder):
    lst_csv = []
    for i in INTERPRETABILITY.keys():
        lst_csv += glob(folder + INTERPRETABILITY[i] + model + '*.csv')
    return lst_csv

def get_id(filename):
    return filename.split('_')[-1].replace('.csv','')

def get_xai_method(filename):
    s = filename.split('_')[-2]
    if s == 'explain':
        s = filename.split('_')[-3]
    return s

def process_data(lst_csv):
    data = {}
    for foo in lst_csv:
        id_sample = get_id(foo)
        method = get_xai_method(foo)
        feat, rank = [], []

        f = open(foo, 'r')
        lines = f.readlines()[1:]
        for i, l in enumerate(lines):
            aux = l.split(',')
            if not 'Sum' in aux[0]:
                feat.append(aux[0])
                rank.append(min(i+1, MAX_FEATURES+1))
        f.close()

        data[str(id_sample)+DELIMITER+method] = pd.DataFrame({'feature': feat, 'rank': rank})

    return data

def plot_global(df, method, out_file):
    labels = df.iloc[:,0].to_numpy()
    attrs = df.iloc[:,1].to_numpy()
    errors = df.iloc[:,2].to_numpy()
    
    plt.xticks(rotation=45)
    barlist = plt.bar(labels, attrs, yerr=errors, error_kw=dict(lw=0.75, capsize=1), align='center', alpha=0.5)
    plt.title('Consensus ranking. {}. {}'.format(method, ','.join(INTERPRETABILITY.keys())))
    plt.tight_layout()

    for i in range(len(labels)):
        if 'Sum' in labels[i]:
            barlist[i].set_color('r')

    plt.savefig(out_file)
    plt.close()

def plot_individual_tables(lst_df, dir_out, model):
    uniq_ids = set([x.split(DELIMITER)[0] for x in lst_df.keys()])

    for id_ in uniq_ids:
        keys = list(filter(lambda x: x.startswith(id_ + DELIMITER), lst_df.keys()))
        df = None
        for k in keys:
            method = k.split(DELIMITER)[-1]
            if df is None:
                df = lst_df[k].copy()
                df.columns = ['feature', method]
            else:
                df = pd.merge(df, lst_df[k].copy(), on='feature', how='inner')
                df = df.rename(columns={'rank': method})

        # add the average ranking
        col = df.iloc[:,1:]
        df['rank'] = col.mean(axis=1)
        df = df.sort_values(by='rank', axis=0)
        df = df.round({'rank': 2})

        # bulk the entire table into a csv file
        df.to_csv('{}/sample_{}_{}.csv'.format(dir_out, model, id_), index=False)

def add_missing_features(df, features):
    df_feats = df['feature'].to_numpy()
    for f in features:
        if f not in df_feats:
            df = df.append({'feature': f, 'rank': MAX_FEATURES+1}, ignore_index=True)
    return df

def summarize(df, key='mean'):
    df2 = df.copy()

    if len(df2) > MAX_FEATURES:
        n_others = len(df2) - MAX_FEATURES
        title = 'Sum other {} features'.format(str(n_others))
        df_others = pd.DataFrame(data=[[title, df2[MAX_FEATURES:][key].mean(), df2[MAX_FEATURES:]['std'].mean()]], columns=['feature', key, 'std'])
        df2 = pd.concat([df2[:MAX_FEATURES], df_others], ignore_index=True)

    return df2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Folder where the files are', required=True)
    args = parser.parse_args()

    dir_out = join(args.folder, FOLDER_OUT)
    create_dir(dir_out)

    for m in MODELS:
        lst_csv = search_files(m, args.folder)
        if len(lst_csv) == 0:
            continue

        dfs = process_data(lst_csv)

        # add mising data: MAX_FEATURES+1
        n_features = max([len(x) for x in dfs.values()])
        features = list(filter(lambda x: len(x) == n_features, dfs.values()))[0]['feature'].to_numpy()

        for d in dfs.values():
            if len(d) < n_features:
                d = add_missing_features(d, features)

        # plot individual samples
        plot_individual_tables(dfs, dir_out, m)
        
        # average data and compute error and standard deviation
        df = pd.concat(dfs.values(), ignore_index=True)
        df_agg = df.groupby('feature').agg(['mean','std']).reset_index()
        df_agg = df_agg.sort_values(by=('rank','mean'), axis=0)
        
        # plot global ranking
        df_agg.columns = ['feature','mean','std']
        df_agg = summarize(df_agg)
        
        plot_global(df_agg, m, '{}/consensus_rank_{}.png'.format(dir_out, m))
        df_agg.to_csv('{}/consensus_rank_{}.csv'.format(dir_out, m), index=False)
