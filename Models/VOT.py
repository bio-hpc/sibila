import os
import json
import joblib
import pickle
import logging
from os.path import join
from typing import Dict, Any
import numpy as np
from .BaseModel import BaseModel
from sklearn.ensemble import VotingClassifier, VotingRegressor
from Tools.TransformResume import TransformResume

   
# Configure logging
logging.basicConfig(
    filename="vot_execution_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define the prefix for VOT
PREFIX_OUT_VOT = "{}_{}"  # Model, Dataset

class VOT(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super().__init__(io_data, cfg, id_list)

        # Ruta para cargar el archivo de configuración
        config_path = os.path.join(
            os.getcwd(), "Common", "Config", "DefaultConfigs", "VOT.json"
        )
        logging.info(f"Cargando configuración desde: {config_path}")
        with open(config_path, "r") as config_file:
            self.vot_config = json.load(config_file)

        # Inicializar atributos desde el archivo de configuración
        self.trained_models_dir = cfg.get_args()["folder"]
        self.task = self.vot_config["type_ml"]
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
                logging.warning(f"Extensión no válida: {file}")
                continue
            try:
                model_name = file.split("_")[0]
                if model_name not in self.vot_config.get("base_models", []):
                    logging.info(f"Modelo {model_name} no está en base_models.")
                    continue
                models[model_name] = loaders[ext](os.path.join(self.trained_models_dir, file))
            except Exception as e:
                logging.error(f"Error cargando modelo {file}: {e}")
        if not models:
            raise ValueError("No se cargaron modelos válidos.")
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
            logging.info(f"Archivo de métricas transformado y guardado en: {transformed_file}")
        except Exception as e:
            logging.warning(f"No se pudo transformar métricas desde ficheros *resume.txt: {e}")
            # Usar pesos por defecto si falla
            default_weight = 1 / len(self.models)
            weights = {model: default_weight for model in self.models.keys()}
            logging.info(f"Pesos por defecto asignados: {weights}")
            return weights

        # Cargar las métricas desde el archivo JSON transformado
        try:
            with open(transformed_file, "r") as f:
                metrics_data = json.load(f)
        except Exception as e:
            logging.error(f"Error al cargar métricas desde el archivo transformado: {e}")
            raise ValueError("No se pudo cargar el archivo de métricas transformado.")

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
                logging.warning(f"No se encontraron métricas para {model_name}. Usando puntaje por defecto.")
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
            logging.info(f"Peso final para {model_name}: {weight:.4f}")

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
                logging.info(f"Predicciones generadas por {model_name}: {pred[:5]} (tipo: {type(pred)})")
            except Exception as e:
                logging.error(f"Error en predicción con {model_name}: {e}")

        if not predictions:
            raise ValueError("No se generaron predicciones válidas de los modelos base.")

        # Convertir listas a arrays
        predictions = np.array(predictions)
        weights = np.array(weights)

        logging.info(f"Formas de predicciones: {[pred.shape for pred in predictions]}")
        logging.info(f"Pesos originales utilizados: {weights}")

        # Llamar al método correspondiente según la tarea
        if self.task == "classification":
            return self._weighted_classification(predictions, weights)
        elif self.task == "regression":
            return self._weighted_regression(predictions, weights)
        else:
            raise ValueError(f"Tarea no soportada: {self.task}")

    def _weighted_classification(self, predictions, weights):
        n_classes = int(np.max(predictions)) + 1
        logging.info(f"Número de clases detectadas: {n_classes}")

        def weighted_vote(x):
            bincount = np.bincount(x, weights=weights[:len(x)], minlength=n_classes)
            return np.pad(bincount, (0, n_classes - len(bincount)), constant_values=0)

        weighted_votes = np.apply_along_axis(weighted_vote, axis=0, arr=predictions.astype(int))
        logging.info(f"Votos ponderados: {weighted_votes}")
        return np.argmax(weighted_votes, axis=0)

    def _weighted_regression(self, predictions, weights):
        """
        Combina las predicciones de regresión de los modelos base utilizando un promedio ponderado.
        """
        logging.info("Iniciando combinación ponderada para regresión.")

        # Convertir las predicciones y los pesos a numpy arrays
        predictions = np.array(predictions)
        weights = np.array(weights)

        if len(predictions) == 0 or len(weights) == 0:
            raise ValueError("No hay predicciones o pesos válidos para procesar en regresión.")

        if len(weights) != predictions.shape[0]:
            raise ValueError("El número de pesos no coincide con el número de modelos.")

        # Calcular el promedio ponderado
        try:
            result = np.average(predictions, axis=0, weights=weights)
            logging.info(f"Resultado ponderado para regresión: {result}")
            return result
        except Exception as e:
            logging.error(f"Error al calcular el promedio ponderado: {e}")
            raise ValueError("Error al combinar predicciones para regresión.")


    def train(self, xtr, ytr):
        """
        Entrena los modelos base y prepara el modelo de votación.
        """
        if xtr is None or ytr is None or len(xtr) == 0 or len(ytr) == 0:
            logging.error("Los datos de entrenamiento (xtr, ytr) no son válidos.")
            raise ValueError("Los datos de entrenamiento no pueden estar vacíos.")

        logging.info(f"Dimensiones de xtr: {xtr.shape if hasattr(xtr, 'shape') else len(xtr)}")
        logging.info(f"Dimensiones de ytr: {len(ytr)}")

        logging.info(f"Modelos base cargados: {list(self.models.keys())}")
        if not self.models:
            raise ValueError("No se han cargado modelos base. Verifique el directorio y la configuración.")

        # Llamar a model_fit para entrenar los modelos base
        self.model_fit(xtr, ytr)

    
    def model_fit(self, xtr, ytr):
        """
        Ajusta los modelos base y prepara el modelo de votación.
        """
        self.io_data.print_m(f"\n\tStart Train {self.cfg.get_params()['model']}")

        # Verificar y guardar las clases objetivo
        self.targets = np.unique(ytr).astype(str)
        logging.info(f"Clases detectadas: {self.targets}")

        # Usar los pesos ya cargados
        if not self.model_weights:
            logging.warning("Pesos no definidos en 'load_interpretability'. Usando pesos iguales por defecto.")
            self.model_weights = {k: 1 / len(self.models) for k in self.models.keys()}
        logging.info(f"Pesos utilizados en 'model_fit': {self.model_weights}")

        # Configurar VotingClassifier
        estimators = [(name, model) for name, model in self.models.items()]
        for name, estimator in estimators:
            logging.info(f"Clasificador: {name}, Tipo: {type(estimator)}")

        # Cambiar a 'hard' si no usas probabilidades (soft requiere predict_proba)
        if self.task == "regression":
            self.model = VotingRegressor(estimators=estimators)
        else:
            self.model = VotingClassifier(estimators=estimators, voting='soft')
        logging.info("Modelo de votación configurado.")

        # Entrenar el VotingClassifier
        try:
            self.model.fit(xtr, ytr)
            logging.info("VotingClassifier entrenado correctamente.")
        except Exception as e:
            logging.error(f"Error al entrenar el VotingClassifier: {e}")
            raise ValueError("Fallo el entrenamiento del VotingClassifier.")

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
                logging.warning(f"El modelo {model_name} no tiene soporte para predict_proba. Detalles: {e}")
                continue  # Opcionalmente, podrías decidir omitir este modelo

            probas.append(proba)

        if not probas:
            raise ValueError("No se generaron probabilidades válidas de los modelos base.")

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
            logging.info(f"Archivo innecesario eliminado: {legacy_path}")
            
        model_path = self.get_prefix()
        logging.info(f"Guardando modelo en: {model_path}")
        joblib.dump(self, model_path)
