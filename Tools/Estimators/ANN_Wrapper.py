class ANN_Wrapper:
    """
    Clase para encapsular el modelo ANN y adaptarlo a las interfaces de scikit-learn.
    """
    def __init__(self, ann_model):
        """
        Inicializa el envoltorio con una instancia de ANN.
        """
        self.ann_model = ann_model

    def fit(self, X, y):
        """
        Entrena el modelo ANN usando el método `train` de la clase ANN.
        """
        self.ann_model.train(X, y)
        return self

    def predict(self, X):
        """
        Realiza predicciones utilizando el método `predict` de la clase ANN.
        """
        return self.ann_model.predict(X)

    def predict_proba(self, X):
        """
        Predice probabilidades para clasificación utilizando la lógica interna de ANN.
        """
        # Aquí usamos directamente el método ANN para `predict`.
        pred = self.ann_model.predict(X)
        # Si las probabilidades ya están definidas como salida, no es necesario ajustar.
        return pred
