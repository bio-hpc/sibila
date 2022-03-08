"""
    LimeRuleExtractor.py:

    Figures out a general rule from the local explanations.
"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import argparse
import pandas as pd
from glob import glob
from os import getcwd
from os.path import isdir, join

class LimeRuleExtractor():
    def __init__(self, folder):
        self.folder = join(getcwd(), folder)
        self.pattern = '*_lime_explain_*'

    def extract_rule(self):
        # list all local interpretations in directory
        files = self.list_files(self.pattern)
        # load data and join it in one single df per target class with columns: feature, condition, weight
        dfs = self.load_data(files)        
        # count the number of appearances of each condition
        # calculate the contribution coefficient for every pair (feature, condition) as: appearances*weight
        result = []
        for df in dfs:
            df_aux = df.groupby(['range','class']).agg(
                count_col=pd.NamedAgg(column='range', aggfunc='count'),
                weigth_mean=pd.NamedAgg(column='weight', aggfunc='mean')
            )
            df_aux['coeff'] = df_aux['count_col']*df_aux['weigth_mean']
            target_id = df_aux.index[0][1]
            result.append((target_id, df_aux.copy()))

        # print all the conditions and coefficients in descending order of importance
        self.print_r(result)

    def list_files(self, pattern):
        files = []
        for name in glob(self.folder + pattern):
            files.append(name)
        return files

    def load_data(self, files):
        df = pd.DataFrame()
        for foo in files:
            df_foo = pd.read_csv(foo, header=0)
            df = df.append(df_foo)
            del df_foo
        
        targets = df['class'].unique()
        df_list = [ df.loc[df['class'] == t] for t in targets ]
        return df_list

    def print_r(self, results):
        for r in results:
            name, df = r[0], r[1]
            df_aux = df.reindex(df['coeff'].abs().sort_values(ascending=False).index)

            print('Identified rules for class {}:'.format(name))
            print(df_aux['coeff'][:10])
            print()
            del df_aux

def dir_path(string):
    if isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Folder containing the local explanations', required=True, type=dir_path)

    args = parser.parse_args()
    ext = LimeRuleExtractor(args.folder)
    ext.extract_rule()
