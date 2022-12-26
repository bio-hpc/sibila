#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jorge de la Peña García"
__version__ = "1.0"
__maintainer__ = "Jorge"
__email__ = "jpena@ucam.edu"
__status__ = "Production"

import os
from os.path import join, splitext, basename, isfile
import tarfile
import shutil
import json
import glob
import pickle


class IOData:
    PATH_DEFAULT_CONFIGS = join('Common', 'Config', 'DefaultConfigs', '*')

    def check_file_exits(self, file):
        if isfile(file):
            return True
        else:
            self.print_e("File {} does not exist")

    def save_dataframe_cols(self, df, cols, file_out):
        if 'weight' in df.columns:
            df = df.reindex(df['weight'].abs().sort_values(ascending=False).index)
        elif 'mean' in df.columns:
            df = df.reindex(df['mean'].abs().sort_values(ascending=False).index)
        df[cols].to_csv(file_out, index=False)

    def read_json(self, file_json):
        self.check_file_exits(file_json)
        with open(file_json) as json_file:
            data = json.load(json_file)
        return data

    def get_file_resume(self):
        return self.file_resume

    def set_file_resume(self, f_resume):
        self.file_resume = f_resume

    def get_lime_folder(self):
        return self.lime_folder

    def set_lime_folder(self, lime_folder):
        self.lime_folder = lime_folder

    def get_integrated_gradients_folder(self):
        return self.integrated_gradients_folder

    def set_integrated_gradients_folder(self, integrated_gradients_folder):
        self.integrated_gradients_folder = integrated_gradients_folder

    def set_dice_folder(self, dice_folder):
        self.dice_folder = dice_folder

    def get_dice_folder(self):
        return self.dice_folder

    def get_shapley_folder(self):
        return self.shapley_folder

    def set_shapley_folder(self, shapley_folder):
        self.shapley_folder = shapley_folder

    def get_pdp_folder(self):
        return self.pdp_folder

    def set_pdp_folder(self, pdp_folder):
        self.pdp_folder = pdp_folder

    def get_ale_folder(self):
        return self.ale_folder

    def set_ale_folder(self, ale_folder):
        self.ale_folder = ale_folder

    def get_anchor_folder(self):
        return self.anchor_folder

    def set_anchor_folder(self, anchor_folder):
        self.anchor_folder = anchor_folder

    def get_job_folder(self):
        return self.job_folder

    def set_job_folder(self, job_folder):
        self.job_folder = job_folder

    def create_dirs(self, folder):
        folder = '{}/'.format(folder) if folder[:-1] != "/" else folder
        self.set_file_resume(folder + "Experiment_out.txt")

        #io_data.create_dir(args.folder + "/Dataset/")
        self.create_dir(folder)
        self.create_dir(folder + "/Dataset/")

        self.set_lime_folder(folder + "LIME/")
        self.create_dir(self.get_lime_folder())

        self.set_integrated_gradients_folder(folder + "Integrated_Gradients/")
        self.create_dir(self.get_integrated_gradients_folder())
        self.create_dir(self.get_integrated_gradients_folder() + "csv/")
        self.create_dir(self.get_integrated_gradients_folder() + "png/")

        self.set_dice_folder(folder + "DICE/")
        self.create_dir(self.get_dice_folder())
        self.create_dir(self.get_dice_folder() + "csv/")
        self.create_dir(self.get_dice_folder() + "png/")

        self.set_shapley_folder(folder + "Shapley/")
        self.create_dir(self.get_shapley_folder())
        self.create_dir(self.get_shapley_folder() + "csv/")
        self.create_dir(self.get_shapley_folder() + "png/")

        self.set_pdp_folder(folder + "PDP/")
        self.create_dir(self.get_pdp_folder())

        self.set_ale_folder(folder + "ALE/")
        self.create_dir(self.get_ale_folder())

        self.set_anchor_folder(folder + "Anchor/")
        self.create_dir(self.get_anchor_folder())

        self.set_job_folder(folder + "jobs/")
        self.create_dir(self.get_job_folder())

    def create_dirs_no_remove(self, folder):
        folder = '{}/'.format(folder) if folder[:-1] != "/" else folder
        self.set_file_resume(folder + "Experiment_out.txt")

        # io_data.create_dir(args.folder + "/Dataset/")
        self.create_dir_no_remove(folder)
        self.create_dir_no_remove(folder + "/Dataset/")

        self.set_lime_folder(folder + "LIME/")
        self.create_dir_no_remove(self.get_lime_folder())

        self.set_integrated_gradients_folder(folder + "Integrated_Gradients/")
        self.create_dir_no_remove(self.get_integrated_gradients_folder())
        self.create_dir_no_remove(self.get_integrated_gradients_folder() + "csv/")
        self.create_dir_no_remove(self.get_integrated_gradients_folder() + "png/")

        self.set_dice_folder(folder + "DICE/")
        self.create_dir(self.get_dice_folder())
        self.create_dir(self.get_dice_folder() + "csv/")
        self.create_dir(self.get_dice_folder() + "png/")

        self.set_shapley_folder(folder + "Shapley/")
        self.create_dir_no_remove(self.get_shapley_folder())
        self.create_dir_no_remove(self.get_shapley_folder() + "csv/")
        self.create_dir_no_remove(self.get_shapley_folder() + "png/")

        self.set_pdp_folder(folder + "PDP/")
        self.create_dir_no_remove(self.get_pdp_folder())

        self.set_ale_folder(folder + "ALE/")
        self.create_dir_no_remove(self.get_ale_folder())

        self.set_anchor_folder(folder + "Anchor/")
        self.create_dir_no_remove(self.get_anchor_folder())

        self.set_job_folder(folder + "jobs/")
        self.create_dir_no_remove(self.get_job_folder())

    def create_dir_no_remove(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    def create_dir(self, folder):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            os.mkdir(folder)

    @staticmethod
    def print_e(txt):
        print("Error: {}\n".format(txt))
        exit()

    def print_m(self, txt):
        t = "\t{}".format(txt)
        print(t)
        f = open(self.get_file_resume(), 'a')
        f.write('{}\n'.format(t))
        f.close()

    def read_all_options(self):
        lst_options = []
        for file_option in glob.glob(self.PATH_DEFAULT_CONFIGS):
            lst_options.append(splitext(basename(file_option))[0].upper())
        return (lst_options)

    def fix_filename(self, str):
        remove_characters = ['/', '<', '>', ':']
        for c in remove_characters:
            str = str.replace(c, '_')
        return str

    def save_time(self, time, file_name):
        f = open(file_name, 'a')
        f.write('{}\n'.format(time))
        f.close()


def filter_function(tarinfo):
   EXCLUDE_DIRS = ['jobs', 'keras_tuner_dir']

   if os.path.isdir(tarinfo.name) and os.path.basename(os.path.normpath(tarinfo.name)) in EXCLUDE_DIRS:
        return None
   return tarinfo


def make_tarfile(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir), filter=filter_function)


def serialize_class(c, serializa_file):
    pickle.dump(c, open(serializa_file, 'wb'))


def get_serialized_params(file_name):
    if isfile(file_name):
        return pickle.load(open(file_name, "rb"))
    else:
        print("ERROR: File not found {}".format(file_name))
        exit()
