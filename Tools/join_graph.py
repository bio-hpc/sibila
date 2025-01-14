import sys
import os
from os.path import basename
sys.path.append(os.path.abspath('.'))
from Common.Analysis.JoinGraphs import JoinGraphs
import argparse
from glob import glob


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=dir_path, help="Folder of experiment")
    args = parser.parse_args()
    jg = JoinGraphs(args.dir)
    jg.create_global()
    jg.join_all_graphs()
