import numpy as np
import tensorflow as tf

class SklearnNetwork(tf.keras.Model):
    def __init__(self, model, input_dim, task="regression", n_classes=None):
        super().__init__()
        self.original_model = model
        self.input_dim = input_dim
        self.task = task
        self.n_classes = n_classes
        self.surrogate = None

    def is_differentiable(self):
        return isinstance(self.original_model, tf.keras.Model)

    def infer_n_classes(self, X):
        """
        auto-detection of the number of classes
        """
        if hasattr(self.original_model, "predict_proba"):
            probs = self.original_model.predict_proba(X[:10])
            return probs.shape[1]
        else:
            y = self.original_model.predict(X[:100])
            return len(np.unique(y))

    def build_surrogate(self, hidden_units=[64, 32]):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        for units in hidden_units:
            model.add(tf.keras.layers.Dense(units, activation="relu"))

        # set the output depending on the type
        if self.task == "classification":
            if self.n_classes is None:
                raise ValueError("n_classes no definido para clasificación")

            if self.n_classes == 2:
                model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
                loss = "binary_crossentropy"
            else:
                model.add(tf.keras.layers.Dense(self.n_classes, activation="softmax"))
                loss = "sparse_categorical_crossentropy"

        else:
            model.add(tf.keras.layers.Dense(1))
            loss = "mse"

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def fit_surrogate(self, X, epochs=100, batch_size=32, verbose=0):
        # auto-detection of classes in case they're not supplied
        if self.task == "classification" and self.n_classes is None:
            self.n_classes = self.infer_n_classes(X)

        # get targets
        y = self.original_model.predict(X) 

        self.surrogate = self.build_surrogate()
        self.surrogate.fit(X, y,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=verbose)

    def call(self, inputs):
        if self.is_differentiable():
            return self.original_model(inputs)
        else:
            return self.surrogate(inputs)

    def prepare(self, X_train):
        if not self.is_differentiable():
            self.fit_surrogate(X_train)
        return self
