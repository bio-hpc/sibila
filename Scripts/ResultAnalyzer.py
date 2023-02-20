#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ResultAnalyzer.py:
    Translates the output files into an Excel with all the metrics.

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import argparse
import codecs
import glob
import json
import numpy as np
import pandas as pd
import tarfile
from os.path import basename, join, isdir
from pathlib import Path
 
class ResultAnalyzer():
	CLASSIFICATION_COLS = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'AUC']
	REGRESSION_COLS = ['Pearson', 'Determination', 'Mean Average Precision', 'MAE', 'MSE']

	def __init__(self, input_dir, output_file):
		self.input_dir = input_dir
		self.output_file = output_file

	def __print(self, msg):
		print(msg)
		print()
	
	def analyse(self):
		self.__print('Scanning directory: %s' % self.input_dir)

		self.__print('Searching for summary files')
		resume = []
		for path in Path(self.input_dir).rglob('*.tar.gz'):
			resume.append(str(path)) 

		if len(resume) == 0:
			self.__print('No results were found')
			exit()

		self.__print('Extracting summary from files')
		raw_values = []
		for datafile in resume:
			raw_values = raw_values + self.__extract_resume(datafile)

		self.__print('Grouping metrics by run name')
		grouped_values = self.__group_metrics(raw_values)

		self.__print('Calculating metrics by test run')
		if self.regression:
			df = self.__calculate_metrics(grouped_values, self.REGRESSION_COLS)
			df = df.sort_values('Determination mean', ascending=False)
		else:
			df = self.__calculate_metrics(grouped_values, self.CLASSIFICATION_COLS)
			df = df.sort_values('AUC mean', ascending=False)

		self.__print('Exporting to Excel')
		self.__export_excel(df)

	def __extract_resume(self, tarfoo):
		data = []

		tar = tarfile.open(tarfoo, "r:gz")
		utf8reader = codecs.getreader('utf-8')
		for name in tar.getmembers():
			if name.name.endswith('_data.json'):
				content = json.load(tar.extractfile(name))
				metrics =  self.__find_metrics(content)
				key = self.__get_key(tarfoo)
				data.append((key, metrics))
		return data

	def __find_metrics(self, data):
		analysis = data['Analysis']

		self.regression = False if 'Auc' in analysis.keys() else True

		if self.regression:
			pearson, pearson_p = [float(x) for x in analysis['Pearson Correlation Coefficient'].split('/')]
			r2 = analysis['Coefficient of Determination']
			avg_precision = analysis['Mean average precision']
			mae = analysis['Mean Absolute Error']
			mse = analysis['Mean Squared Error']
			return pearson, r2, avg_precision, mae, mse
		else:
			accuracy = analysis['Accuracy']
			precision = analysis['Precision']
			f1 = analysis['F1']
			recall = analysis['Recall']
			specificity = analysis['Specificity']
			auc = analysis['Auc']
			return accuracy, precision, f1, recall, specificity, auc

	def __get_key(self, filename):
		return basename(filename).split('_')[0]

	def __group_metrics(self, flat_values):
		d = {}
		for v in flat_values:
			key, metrics = v[0], v[1]
			if key not in d:
				d[key] = [ metrics ]
			else:
				d[key].append(metrics)
		return d

	def __calculate_metrics(self, data, col_names):
		columns = ['Model']
		for c in col_names:
			columns += [c+' mean'] + [c+' std']

		L = [(k, *t) for k, v in data.items() for t in v]
		df = pd.DataFrame(L, columns=['Model']+col_names)
		df = df.groupby(['Model']).agg({k:['mean','std'] for k in col_names}).fillna(0).reset_index()
		df.columns = columns

		return df

	def __export_excel(self, df):
		df.to_excel(self.output_file, sheet_name='Summary', float_format="%.3f", 
			freeze_panes=(1,1), columns=df.columns, index=False)

def main(args):
	a = ResultAnalyzer(args.dir, output_file=args.output)
	a.analyse()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Summarize the results and export them to Excel.')
	parser.add_argument('-d', '--dir', help='Directory where files are located', required=True)
	parser.add_argument('-o', '--output', help='Output file with extension xlsx', default='Summary.xlsx')

	args = parser.parse_args()
	main(args)
