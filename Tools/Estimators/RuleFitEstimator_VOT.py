#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from rulefit import RuleFit

class RuleFitEstimator_VOT(BaseEstimator, ClassifierMixin):
    def __init__(self, io_data=None, cfg=None, id_list=None):
        """
        Wrapper para RuleFit compatible con VotingClassifier,
        inicializado utilizando configuración basada en `cfg`.

        :param io_data: Datos de entrada (puede no ser necesario en esta implementación).
        :param cfg: Objeto de configuración que contiene los parámetros del modelo.
        :param id_list: Lista de identificadores para las características.
        """
        self.io_data = io_data
        self.cfg = cfg
        self.id_list = id_list
        self._estimator_type = 'classifier'
        self.classes_ = []
        
        # Inicializar RuleFit utilizando configuración de cfg
        if self.cfg:
            params = self.cfg.get_params()['params']
            self.rp_model = RuleFit(**params)
        else:
            raise ValueError("Se requiere un objeto cfg válido para inicializar RuleFitEstimator_VOT.")

    def fit(self, X, Y):
        """
        Ajusta el modelo RuleFit con los datos de entrada.

        :param X: Características de entrenamiento.
        :param Y: Etiquetas de entrenamiento.
        """
        self.rp_model.fit(X, Y)
        self.classes_ = np.unique(Y)

    def predict_proba(self, X):
        """
        Devuelve las probabilidades de predicción.

        :param X: Características de entrada.
        :return: Array de probabilidades.
        """
        try:
            ypr = self.rp_model.predict(X.values)
        except AttributeError:
            ypr = self.rp_model.predict(X)

        # Normalizar y asegurar formato como probabilidades
        ypr = np.round(ypr, 3)
        ypr = np.array(ypr)

        return ypr

    def predict(self, X):
        """
        Predice las etiquetas utilizando un umbral para clasificar.

        :param X: Características de entrada.
        :return: Etiquetas predichas.
        """
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)

    def decision_function(self, X):
        """
        Calcula la función de decisión basada en predict_proba.

        :param X: Características de entrada.
        :return: Valores de la función de decisión.
        """
        return self.predict_proba(X)

    def get_params(self, deep=True):
        """
        Retorna los parámetros del estimador para compatibilidad con scikit-learn.

        :param deep: Si incluir sub-objetos en los parámetros.
        :return: Diccionario de parámetros.
        """
        return {"io_data": self.io_data, "cfg": self.cfg, "id_list": self.id_list}

    def set_params(self, **params):
        """
        Configura los parámetros del estimador.

        :param params: Diccionario de parámetros a configurar.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
