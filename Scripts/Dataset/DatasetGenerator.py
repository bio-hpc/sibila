#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Development"

import argparse
import numpy as np
import pandas as pd

class DatasetGenerator:

    ZERO = 0.0
    ONE = 1.0

    def __init__(self, n_samples, n_features, percentage, rules, out_file='dataset.csv'):
        self.n_samples = n_samples
        self.n_features = n_features
        self.pct_1 = percentage
        self.rules = rules
        self.out_file = out_file
        self.feature_names = [ 'F'+str(i+1) for i in range(n_features) ] + ['class']

    def make_dataset(self):
        data = np.empty((self.n_features+1, self.n_samples))
        rules_store = self.read_rules()

        rows_1 = int(self.n_samples * self.pct_1)
        rows_0 = self.n_samples - rows_1

        for i in range(self.n_features):
            col_no = i + 1
            if col_no in rules_store:
                min_ = rules_store[col_no][0]
                max_ = rules_store[col_no][1]

                if min_ > self.ZERO and max_ < self.ONE:
                    xinf = int(rows_0/2)
                elif min_ == self.ZERO:
                    xinf = 0
                elif max_ == self.ONE:
                    xinf = rows_0

                x1 = np.random.uniform(low=self.ZERO, high=min_, size=(1, xinf))
                x2 = np.random.uniform(low=max_*1.01, high=self.ONE, size=(1, rows_0 - xinf))

                col_1 = np.random.uniform(low=min_, high=max_, size=(1, rows_1))
                col_0 = np.concatenate((x1, x2), axis=1)
                data[i] = np.concatenate((col_1, col_0), axis=1)
            else:
                data[i] = np.random.uniform(low=self.ZERO, high=self.ONE, size=(1, self.n_samples))

        # arrange and set type of target columns
        data = data.T
        data[:rows_1, -1] = 1
        data[rows_1:, -1] = 0
    
        # convert matrix into dataframe and shuffle the data
        df = pd.DataFrame(data, columns=self.feature_names)
        df['class'] = df['class'].astype(int)
        df = df.sample(frac=1).reset_index(drop=True)

        # export
        df.to_csv(self.out_file)
        self.print_summary(df)

    def read_rules(self):
        def parse_rule(rule_str):
            s = rule_str.split(',')
            for i in range(3):
                if i == 0:
                    col_id = self.check_value(s[i])
                    if col_id is None:
                        raise Exception('The number of the rule is required')
                    col_id = int(col_id)
                elif i == 1:
                    min_val = self.check_value(s[i], default=0.0)
                elif i == 2:
                    max_val = self.check_value(s[i], default=1.0)

            return col_id, min_val, max_val

        d = {}
        for r in self.rules:
            a, b, c = parse_rule(r)
            d[a] = (b, c)
        return d

    def check_value(self, value, default=None):
        if value is None or len(value) == 0:
            return default
        return float(value)

    def print_summary(self, df):
        seriesObj = df.apply(lambda x: True if x['class'] == 1 else False , axis=1)
        positive_samples = len(seriesObj[seriesObj == True].index)
        positive_ratio = (positive_samples/len(df))*100.

        print('The dataset has been created in {}'.format(self.out_file))
        print('Number of samples: ', self.n_samples)
        print('Number of features: ', self.n_features)
        print('Number of positive (class 1) samples: ', positive_samples)
        print('Ratio of positive samples: {}%'.format(str(positive_ratio)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a synthetic dataset under the given conditions')
    parser.add_argument('-s', '--samples', type=int, help='Number of samples (rows)', required=True)
    parser.add_argument('-f', '--features', type=int, help='Number of features (columns)', required=True)
    parser.add_argument('-p', '--percentage', type=float, help='Percentage of samples in class 1', default=0.5)
    parser.add_argument('-r', '--rules', type=str, help='Rules', nargs='+')
    parser.add_argument('-o', '--output', type=str, help='Output file', default='dataset.csv')
    args = parser.parse_args()

    dg = DatasetGenerator(args.samples, args.features, args.percentage, args.rules, args.output)
    dg.make_dataset()
