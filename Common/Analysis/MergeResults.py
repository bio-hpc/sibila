#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MergeResults.py:
    Gather all the information from the json files obtained in the tests and create a table in a csv file
"""
__author__ = "Jorge de la Peña García"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jpena@ucam.edu"
__status__ = "Production"

import glob
import json
from os.path import splitext, basename, join


class MergeResults:
    F_RESUME_CSV = 'Resume_analysis.csv'
    SEPARATOR_CSV = ';'
    MAX_TEST = 100  # maximum number of tests allowed per model

    def __init__(self, folder_experiment):
        self.folder = folder_experiment

        self.results = {}
        self.create_csv()

    def get_key_name(self, filename):
        key_name = splitext(basename(filename))[0].split('_')[0]

        if key_name in self.results:
            for i in range(2, self.MAX_TEST):
                if key_name+"_"+str(i) not in self.results:
                    key_name = key_name+"_"+str(i)
                    break;
            #print('Error: ya existe un modelo con ese nombre')
            #exit()
        return key_name

    def read_data(self):

        for i in glob.glob(join(self.folder, '*.json')):
            key_name = self.get_key_name(i)
            with open(i) as file_in:
                self.results[key_name] = json.load(file_in)

    def create_csv(self):
        self.read_data()
        csv_header = list(self.results.keys())
        csv_header.sort()

        lst_lines = [self.SEPARATOR_CSV + '{} '.format(self.SEPARATOR_CSV).join(csv_header)]
        lst_lines += self.analysis(csv_header)
        lst_lines.append("")
        lst_lines += self.config(csv_header)

        f = open(join(self.folder, self.F_RESUME_CSV), 'w')
        for i in lst_lines:
            f.write('{}\n'.format(i))
        f.close()

    def config(self, csv_header):

        lst_keys_config = []
        lst_lines = []
        for i in csv_header:
            for k, v in self.results[i]['Config']['Model_params']['params'].items():
                if k not in lst_keys_config:
                    lst_keys_config.append(k)

        for k in lst_keys_config:
            line = k
            for key in csv_header:
                if k in self.results[key]['Config']['Model_params']['params']:
                    line += self.SEPARATOR_CSV+" " + str(self.results[key]['Config']['Model_params']['params'][k])
                else:
                    line += self.SEPARATOR_CSV + "-"
            lst_lines.append(line)
        return lst_lines


    def analysis(self, csv_header):
        lst_lines = []
        keys_analysis = self.results[csv_header[0]]['Analysis'].keys()
        for key in keys_analysis:
            if key != "Confusion matrix":
                line = key
                for header in csv_header:
                    line += self.SEPARATOR_CSV + ' ' + str(self.results[header]['Analysis'][key]).replace('.',',')
                # print(line)
                lst_lines.append(line)
        return lst_lines
