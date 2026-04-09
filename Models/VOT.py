import os
import json
import joblib
import pickle
from os.path import join
from typing import Dict, Any
import numpy as np
from .BaseModel import BaseModel
from sklearn.ensemble import VotingClassifier, VotingRegressor
from Tools.TransformResume import TransformResume


# Define the prefix for VOT
PREFIX_OUT_VOT = "{}_{}"  # Model, Dataset

class VOT(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super().__init__(io_data, cfg, id_list)

        # Ruta para cargar el archivo de configuración
        config_path = os.path.join(
            os.getcwd(), "Common", "Config", "DefaultConfigs", "VOT.json"
        )
        self.io_data.print_m(f"Loading config from: {config_path}")
        with open(config_path, "r") as config_file:
            self.vot_config = json.load(config_file)

        # Inicializar atributos desde el archivo de configuración
        self.trained_models_dir = cfg.get_args()["folder"]
        self.task = "regression" if cfg.get_args()["regression"] else "classification"
        self.remove_outliers = self.vot_config.get("remove_outliers", False)

        # Cargar modelos base y sus datos asociados
        self.models = self.load_models()
        
        # Cargar pesos de interpretabilidad
        self.model_weights = self.load_evaluation_weights()
        
        estimators = [(name, model) for name, model in self.models.items()]
        if self.task == "regression":
            self.model = VotingRegressor(estimators=estimators)
        else:
            # Cambiar a 'hard' si no se usa probabilidades
            self.model = VotingClassifier(estimators=estimators, voting='soft')  
        
    def load_models(self) -> Dict[str, Any]:
        loaders = {".joblib": joblib.load, ".dat": lambda p: pickle.load(open(p, "rb"))}
        models = {}
        for file in os.listdir(self.trained_models_dir):
            _, ext = os.path.splitext(file)
            if ext not in loaders:
                self.io_data.print_m(f"Invalid extension: {file}")
                continue
            try:
                model_name = file.split("_")[0]
                if model_name not in self.vot_config.get("base_models", []):
                    self.io_data.print_m(f"Model {model_name} is not available.")
                    continue
                models[model_name] = loaders[ext](os.path.join(self.trained_models_dir, file))
            except Exception as e:
                self.io_data.print_e(f"Error loading model {file}: {e}")
        if not models:
            raise ValueError("No valid models were loaded.")
        return models



    def load_evaluation_weights(self):
        """
        Carga los pesos de interpretabilidad desde un archivo transformado usando transform_resume.
        Si el archivo no existe, asigna pesos por defecto.
        """
        weights = {}

        # Crear subcarpeta /tmp dentro del directorio de modelos si no existe
        tmp_dir = os.path.join(self.trained_models_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Ruta del archivo transformado dentro de /tmp
        transformed_file = os.path.join(tmp_dir, "transformed_resume.json")

        # Usar TransformResume para procesar todos los *_resume.txt generados por los modelos base
        try:
            TransformResume(self.trained_models_dir, transformed_file)
            self.io_data.print_m(f"Metrics file transformed and saved in: {transformed_file}")
        except Exception as e:
            self.io_data.print_m(f"Metrics could not be loaded from *resume.txt: {e}")
            # Usar pesos por defecto si falla
            default_weight = 1 / len(self.models)
            weights = {model: default_weight for model in self.models.keys()}
            self.io_data.print_m(f"Default weights assigned: {weights}")
            return weights

        # Cargar las métricas desde el archivo JSON transformado
        try:
            with open(transformed_file, "r") as f:
                metrics_data = json.load(f)
        except Exception as e:
            self.io_data.print_e(f"Error loading metrics from transformed file: {e}")
            raise ValueError("Could not load metrics from transformed file.")

        # Métricas y sus pesos relativos
        metric_weights = self.vot_config["metric_weights"]

        # Calcular puntajes de métricas
        model_scores = {}
        for model_name in self.models.keys():
            model_metrics = [
                item for item in metrics_data if item["Model"] == model_name and item["Metric"] in metric_weights
            ]

            if model_metrics:
                # Calcular el puntaje como suma ponderada de las métricas
                total_score = sum(
                    metric_weights[item["Metric"]] * item["Value"] for item in model_metrics
                )
                model_scores[model_name] = total_score
            else:
                self.io_data.print_m(f"Metrics not found for {model_name}. Use default scoring.")
                model_scores[model_name] = 0  # Penalización directa para modelos sin métricas

        # Generar un ranking basado en los puntajes
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}

        # Asignar pesos basados en el ranking
        weights = {model: 1 / rank for model, rank in ranking.items()}

        # Normalizar los pesos para que sumen 1
        total_weight_sum = sum(weights.values())
        weights = {k: v / total_weight_sum for k, v in weights.items()}

        # Log de pesos finales
        for model_name, weight in weights.items():
            self.io_data.print_m(f"Final weight for {model_name}: {weight:.4f}")

        return weights


    def predict(self, xts):
        """
        Realiza predicciones utilizando votación ponderada para clasificación o promedio ponderado para regresión.
        """
        predictions = []
        weights = []

        for model_name, model in self.models.items():
            try:
                pred = model.predict(xts)
                predictions.append(np.array(pred))
                weights.append(self.model_weights.get(model_name, 1))  # Obtener peso del modelo
                self.io_data.print_m(f"Predictions for {model_name}: {pred[:5]} (tipo: {type(pred)})")
            except Exception as e:
                self.io_data.print_e(f"Error in prediction with {model_name}: {e}")

        if not predictions:
            raise ValueError("No valid predictions were generated from base models.")

        # Convertir listas a arrays
        predictions = np.array(predictions)
        weights = np.array(weights)

        self.io_data.print_m(f"Prediction shapes: {[pred.shape for pred in predictions]}")
        self.io_data.print_m(f"Using original weights: {weights}")

        # Llamar al método correspondiente según la tarea
        if self.task == "classification":
            return self._weighted_classification(predictions, weights)
        elif self.task == "regression":
            return self._weighted_regression(predictions, weights)
        else:
            raise ValueError(f"Task not supported: {self.task}")

    def _weighted_classification(self, predictions, weights):
        n_classes = int(np.max(predictions)) + 1
        self.io_data.print_m(f"Number of classes detected: {n_classes}")

        def weighted_vote(x):
            bincount = np.bincount(x, weights=weights[:len(x)], minlength=n_classes)
            return np.pad(bincount, (0, n_classes - len(bincount)), constant_values=0)

        weighted_votes = np.apply_along_axis(weighted_vote, axis=0, arr=predictions.astype(int))
        self.io_data.print_m(f"Weighted votes: {weighted_votes}")
        return np.argmax(weighted_votes, axis=0)

    def _weighted_regression(self, predictions, weights):
        """
        Combina las predicciones de regresión de los modelos base utilizando un promedio ponderado.
        """
        self.io_data.print_m("Starting weighted combination for regression.")

        # Convertir las predicciones y los pesos a numpy arrays
        predictions = np.array(predictions)
        weights = np.array(weights)

        if len(predictions) == 0 or len(weights) == 0:
            raise ValueError("No valid predictions or weights for processing in regression.")

        if len(weights) != predictions.shape[0]:
            raise ValueError("The number of weights does not match the number of models.")

        # Calcular el promedio ponderado
        try:
            result = np.average(predictions, axis=0, weights=weights)
            self.io_data.print_m(f"Result of weighted combination for regression: {result}")
            return result
        except Exception as e:
            raise ValueError("Error combining predictions for regression.")

    def train(self, xtr, ytr):
        """
        Entrena los modelos base y prepara el modelo de votación.
        """
        if xtr is None or ytr is None or len(xtr) == 0 or len(ytr) == 0:
            raise ValueError("Training data cannot be empty.")

        self.io_data.print_m(f"Base models loaded: {list(self.models.keys())}")
        if not self.models:
            raise ValueError("No base models were loaded. Check the directory and configuration.")

        # Llamar a model_fit para entrenar los modelos base
        self.model_fit(xtr, ytr)

    
    def model_fit(self, xtr, ytr):
        """
        Ajusta los modelos base y prepara el modelo de votación.
        """
        self.io_data.print_m(f"\n\tStart Train {self.cfg.get_params()['model']}")

        # Verificar y guardar las clases objetivo
        self.targets = np.unique(ytr).astype(str)
        self.io_data.print_m(f"Detected classes: {self.targets}")

        # Usar los pesos ya cargados
        if not self.model_weights:
            self.io_data.print_m("Undefined weights in 'load_interpretability'. Using default weights.")
            self.model_weights = {k: 1 / len(self.models) for k in self.models.keys()}
        self.io_data.print_m(f"Using weights: {self.model_weights}")

        # Configurar VotingClassifier
        estimators = [(name, model) for name, model in self.models.items()]
        for name, estimator in estimators:
            self.io_data.print_m(f"Classifier: {name}, Type: {type(estimator)}")

        # Cambiar a 'hard' si no usas probabilidades (soft requiere predict_proba)
        if self.task == "regression":
            self.model = VotingRegressor(estimators=estimators)
        else:
            self.model = VotingClassifier(estimators=estimators, voting='soft')
        self.io_data.print_m("VotingClassifier configured.")

        # Entrenar el VotingClassifier
        try:
            self.model.fit(xtr, ytr)
            self.io_data.print_m("VotingClassifier trained successfully.")
        except Exception as e:
            raise ValueError("Failed training of VotingClassifier.")

        self.io_data.print_m(f"End Train {self.cfg.get_params()['model']}")

    
    def predict_proba(self, X):
        """
        Predice probabilidades para el VotingClassifier en VOT.
        Combina las probabilidades de los modelos base, incluso si algunos no tienen soporte nativo para predict_proba.
        """
        probas = []
        for model_name, model in self.models.items():
            try:
                # Usa el método predict_proba heredado de BaseModel
                proba = model.predict_proba(X)
            except AttributeError as e:
                self.io_data.print_m(f"Model {model_name} does not support predict_proba. Details: {e}")
                continue  # Opcionalmente, podrías decidir omitir este modelo

            probas.append(proba)

        if not probas:
            raise ValueError("No valid probabilities were generated from base models.")

        # Combina las probabilidades promediando
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    
    def get_prefix(self):
        """
        Devuelve la ruta para guardar los resultados del modelo VOT.
        """
        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_VOT.format(
                self.cfg.get_params()["model"], self.cfg.get_name_dataset()))

    def save_model(self):
        """
        Guarda el modelo VOT en la ruta especificada por el prefijo.
        """
        # Eliminar archivo innecesario en el directorio raíz si existe
        legacy_path = os.path.join(self.trained_models_dir, "transformed_resume.json")
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
            self.io_data.print_m(f"Removed unnecessary file: {legacy_path}")
            
        model_path = self.get_prefix()
        self.io_data.print_m(f"Model saved in: {model_path}")
        joblib.dump(self, model_path)
