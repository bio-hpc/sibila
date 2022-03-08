import argparse


def read_params():
    parser = argparse.ArgumentParser(description='SIBILA')
    parser.add_argument('csv', type=argparse.FileType('r'))
    args = parser.parse_args()
    return args
