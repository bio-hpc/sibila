from Tools.Graphics import Graphics
import subprocess as sp
import os

class GPUTracker:

    def __init__(self, prefix):
        self.prefix = prefix
        self.logger_fname = f"{prefix}_log_compute.csv"
        self.logger_pid = -1

    def start(self, type_model):
        if os.path.exists(self.logger_fname):
            os.remove(self.logger_fname)

        self.logger_pid = sp.Popen(['python', 'log_gpu_cpu_stats.py', self.logger_fname,'--loop','0.2'])
        print(f'Started logging compute utilisation: {type_model}')

    def stop(self):
        self.logger_pid.kill()

    def plot(self):
        Graphics().plot_gpu_usage(self.logger_fname, f"{self.prefix}_gpu_usage.png")
