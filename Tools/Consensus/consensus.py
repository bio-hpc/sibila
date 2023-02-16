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
from os.path import join, isdir
from ConsensusAverageMean import ConsensusAverageMean
from ConsensusHarmonicMean import ConsensusHarmonicMean
from ConsensusGeometricMean import ConsensusGeometricMean
from ConsensusVoting import ConsensusVoting
from ConsensusAverageRank import ConsensusAverageRank

FOLDER_OUT = "Consensus/"

def create_dir(dir):
    if not isdir(dir):
        os.mkdir(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Folder where the files are', required=True)
    args = parser.parse_args()

    # create the output folder for consensus
    dir_out = join(args.folder, FOLDER_OUT)
    #TODO create_dir(dir_out)

    # call consensus
    c = ConsensusAverageRank(args.folder)
    c.run()

