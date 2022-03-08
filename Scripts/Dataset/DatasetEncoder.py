#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

class DatasetEncoder():
    def __init__(self, filename, separator=None, variance=None, ratio_null=None, target=None, index=None):
    	self.filename = filename
    	self.index = index
    	self.target = target
    	self.separator = separator
    	self.variance = variance
    	self.ratio_null = ratio_null

    def print_s(self, df):
    	print('Shape: ', df.shape)

    def print_i(self, msg):
    	print('INFO: {}......'.format(msg))

    def remove_empty_data(self, df):
        # Remove entirely empty columns
        df = df.dropna(how='all', axis=1)
        # Remove entirely empty rows
        df = df.dropna(how='all', axis=0)
        return df

    def process_target(self, df):
        # Move target to the last position
        target_col = df.pop(self.target)
        df = pd.concat([df, target_col], axis=1)

        # Remove rows without target class
        df.dropna(subset=[self.target], inplace=True)

        # Encode the target with a LabelEncoder
        le = LabelEncoder()
        df[self.target] = le.fit_transform(df[self.target])
        self.print_i('Target classes: {}'.format(le.classes_))

        return df

    def remove_nulls(self, df, ratio_null):
        percent_missing = df.isnull().sum() * 100 / len(df)    	
        for i, c in enumerate(df.columns):
            if c in [self.index, self.target]:
                continue

            if percent_missing[i] >= ratio_null:
                df.drop([c], axis=1, inplace=True)
                self.print_i('The column {} has been removed with {}% of null values'.format(c, round(percent_missing[i],2)))
        return df

    def remove_variance(self, df, variance):
        col_variance = df.var()
        for item in col_variance.iteritems():
            if item[0] in [self.index, self.target]:
                continue

            if item[1] <= variance:
                df.drop([item[0]], axis=1, inplace=True)
                self.print_i('The column {} has been removed because its variance is {}'.format(item[0], round(item[1],2)))
        return df

    def is_date(self, df, col_name):
        def str_to_date(s):
            try:
                x = datetime.strptime(s, '%m/%d/%Y')
                return True
            except Exception as e:
    	        return False

        return df[col_name].dropna().transform(lambda x: str_to_date(x)).all()

    def encode_date(self, df, c):
        df[c] = df[c].dropna().transform(lambda x: datetime.timestamp(datetime.strptime(x, '%m/%d/%Y')))

    def encode(self):
        self.print_i('Loading input file')
        df = pd.read_csv(self.filename, sep=self.separator)
        self.print_i('Initial shape: {}'.format(df.shape))

        if self.index is None:
        	self.index = df.columns[0]
        if self.target is None:
        	self.target = df.columns[-1]

        self.print_i('Index column is "{}"'.format(self.index))
        self.print_i('Target column is "{}"'.format(self.target))

        shape = df.shape
        df = self.remove_empty_data(df)
        self.print_i('Removed {} columns'.format(shape[1]-df.shape[1]))
        self.print_i('Removed {} rows'.format(shape[0]-df.shape[0]))

        # Remove columns with high ratio of null
        df = self.remove_nulls(df, self.ratio_null)

        # Remove columns with low variance
        df = self.remove_variance(df, self.variance)
        
        # Move index to the first position
        df.set_index(self.index)
        
        # Encode categorical columns but the index and target
        self.print_i('\n========== ENCODING ==========\n')
        num_cols = list(set(df._get_numeric_data().columns) - set([self.index, self.target]))
        cat_cols = list(set(df.columns) - set(num_cols) - set([self.index, self.target]))
        
        for c in cat_cols:
            if self.is_date(df, c):
            	self.print_i('Encoding date column "{}"'.format(c))
            	self.encode_date(df, c)
            else:
                self.print_i('Encoding categorical column "{}"'.format(c))
                df = pd.concat([df, pd.get_dummies(df[c], prefix=c, dummy_na=True)], axis=1)
                df.drop([c], axis=1, inplace=True)
        
        # Process the target column
        df = self.process_target(df)
        self.print_i('The target column has been encoded')

        # Fill in empty cells in numerical columns with the average value
        for c in df.columns:
            df[c].fillna((df[c].mean()), inplace=True)
        
        # Export the updated dataset to CSV
        self.print_i('Output shape: {}'.format(df.shape))
        out_filename = self.filename.replace('.csv', '_encoded.csv')
        df.to_csv(out_filename, index=False)
        self.print_i('Dataset exported to {}'.format(out_filename))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Encode a dataset.')
	parser.add_argument('-f', '--file', type=str, help='Input file', required=True)
	parser.add_argument('-t', '--target', type=str, help='Target column', default='Last one')
	parser.add_argument('-i', '--index', type=str, help='Index column', default='First one')
	parser.add_argument('-s', '--separator', type=str, help='Column separator', default=',')
	parser.add_argument('-v', '--variance', type=float, help='Variance under which columns will be removed', default=0.0)
	parser.add_argument('-n', '--null', type=float, help='Ration of nulls above which columns will be removed', default=100.0)
	args = parser.parse_args()

	enc = DatasetEncoder(args.file, separator=args.separator, target=args.target, index=args.index, 
		variance=args.variance, ratio_null=args.null)
	enc.encode()
