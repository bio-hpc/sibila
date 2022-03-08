import tensorflow as tf

class SklearnNetwork(tf.keras.Sequential):

    def __init__(self, model, input_shape):
        super(SklearnNetwork, self).__init__( layers = [ tf.keras.layers.InputLayer(input_shape=(1, input_shape[1])), tf.keras.layers.Dense(1) ] )
        self.model = model

    #def predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
    #    return self.model.predict(x)
