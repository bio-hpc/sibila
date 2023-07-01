import tensorflow as tf
from Tools.Graphics import Graphics

class LearningHistoryCallback(tf.keras.callbacks.Callback):

    # Use class variables to accumulate the effect of cross validation along the entire training
    epochs = 0
    loss = []
    acc = {}

    def __init__(self, cfg):
        super(LearningHistoryCallback, self).__init__()
        self.cfg = cfg
        self.g = Graphics()

    def on_epoch_end(self, epoch, logs=None):
        LearningHistoryCallback.epochs = LearningHistoryCallback.epochs + 1
        for k in logs.keys():
            if k not in LearningHistoryCallback.acc.keys():
                LearningHistoryCallback.acc[k] = []
            LearningHistoryCallback.acc[k].append(logs[k])

    def on_train_end(self, logs=None):
        foo = self.cfg.get_prefix() + "_loss_acc.png"
        self.g.plot_metrics_evolution(LearningHistoryCallback.epochs, LearningHistoryCallback.acc, foo)
