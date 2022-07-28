import tensorflow as tf
from Tools.Graphics import Graphics

class LearningHistoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, cfg):
        super(LearningHistoryCallback, self).__init__()
        self.cfg = cfg
        self.g = Graphics()
        self.epochs = 0
        self.loss = []
        self.acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs = self.epochs + 1
        self.loss.append(logs['loss'])
        self.acc.append(logs['accuracy'])

    def on_train_end(self, logs=None):
        foo = self.cfg.get_prefix() + "_loss_acc.png"
        self.g.plot_metrics_evolution(self.loss, self.acc, self.epochs, foo)
