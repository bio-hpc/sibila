"""
    consensus.py:

    Makes consensus of interpretability for a model

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import argparse
import os
from os.path import exists, join
from ConsensusAverageMean import ConsensusAverageMean
from ConsensusHarmonicMean import ConsensusHarmonicMean
from ConsensusGeometricMean import ConsensusGeometricMean
from ConsensusVoting import ConsensusVoting
from ConsensusAverageRank import ConsensusAverageRank
from ConsensusCustom import ConsensusCustom

FOLDER_OUT = "Consensus/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Folder where the files are', required=True)
    parser.add_argument('-m', '--method', help='Consensus method', required=True, choices=['AM','HM','GM','VF','AR','CU'])
    args = parser.parse_args()

    # create the output folder for consensus
    dir_out = join(args.folder, FOLDER_OUT)
    if not exists(dir_out):
        os.makedirs(dir_out)

    # call consensus
    switch_method = {
        'AM': ConsensusAverageMean(args.folder, dir_out),
        'HM': ConsensusHarmonicMean(args.folder, dir_out),
        'GM': ConsensusGeometricMean(args.folder, dir_out),
        'VF': ConsensusVoting(args.folder, dir_out),
        'AR': ConsensusAverageRank(args.folder, dir_out),
        'CU': ConsensusCustom(args.folder, dir_out)
    }

    c = switch_method.get(args.method)
    if c is None:
        print('ERROR: Option {} does not exist'.format(args.method))
        exit()
    c.run()

