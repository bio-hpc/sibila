import sys
from Tools.IOData import make_tarfile
from Common.Analysis.JoinGraphs import JoinGraphs

class EndProcess:
    def __init__(self, dir_name):
        self.end_proces(dir_name)

    def end_proces(self, dir_name):
        self.join_graphs(dir_name)
        self.compress(dir_name)

    def join_graphs(self, dir_name):
        jg = JoinGraphs(dir_name)
        jg.create_global()
        jg.join_all_graphs()

    def compress(self, dir_name):
        make_tarfile(dir_name, dir_name + ".tar.gz")


if __name__ == "__main__":
    dir_name = sys.argv[1]
    EndProcess(dir_name)
