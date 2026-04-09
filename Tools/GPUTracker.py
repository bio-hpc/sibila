from Tools.Graphics import Graphics
import subprocess as sp
import os
import tensorflow as tf

class GPUTracker:

    def __init__(self, prefix):
        self.prefix = prefix
        self.logger_fname = f"{prefix}_log_compute.csv"
        self.logger_pid = -1
        self.gpus = self.__check_avilable_gpus()
        self.gpu_available = self.__check_gpu_compatibility()

    def start(self, type_model):
        if os.path.exists(self.logger_fname):
            os.remove(self.logger_fname)

        self.logger_pid = sp.Popen(['python', 'log_gpu_cpu_stats.py', self.logger_fname,'--loop','0.2'])
        print(f'Started logging compute utilisation: {type_model}')

    def stop(self):
        self.logger_pid.kill()

    def plot(self):
        Graphics().plot_gpu_usage(self.logger_fname, f"{self.prefix}_gpu_usage.png")

    def __check_avilable_gpus(self):
        return tf.config.list_physical_devices('GPU')

    def __check_gpu_compatibility(self):
        if len(self.gpus) > 0:
            try:
                # this fails is CUDA is not compatible
                with tf.device('/GPU:0'):
                    tf.constant([1.0])
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return True

            except (tf.errors.InternalError, RuntimeError) as e:
                print(f"GPU detected but not functional: {e}.")
                print("Dynamically forcing the use of CPU.")
            
                # dynamically deactivate GPUs for tensorflow
                tf.config.set_visible_devices([], 'GPU')
                return False
        else:
            print("No physical GPUs detected. CPU will be used instead.")
            return False
