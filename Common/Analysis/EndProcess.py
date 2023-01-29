import glob
import sys
from Tools.Graphics import Graphics
from Tools.IOData import make_tarfile, get_serialized_params
from Common.Analysis.JoinGraphs import JoinGraphs

class EndProcess:
    def __init__(self, dir_name):
        self.g = Graphics()
        self.end_proces(dir_name)

    def end_proces(self, dir_name):
        self.make_time_plots(dir_name)
        self.join_graphs(dir_name)
        self.compress(dir_name)

    def make_time_plots(self, dir_name):
        lst = glob.glob(dir_name + '/*.pkl')
        for foo in lst:
            params = get_serialized_params(foo).get_params()
            cfg = params['cfg']
            self.g.plot_times(cfg.get_folder(), cfg.get_prefix(), cfg.get_params()['model'])

    def join_graphs(self, dir_name):
        jg = JoinGraphs(dir_name)
        jg.create_global()
        jg.join_all_graphs()

    def compress(self, dir_name):
        make_tarfile(dir_name, dir_name + ".tar.gz")

if __name__ == "__main__":
    dir_name = sys.argv[1]
    EndProcess(dir_name)
