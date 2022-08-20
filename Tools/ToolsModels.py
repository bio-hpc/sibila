import tensorflow as tf
from Tools.TypeML import TypeML
from Tools.ClassFactory import ClassFactory
from Tools.Estimators.RipperEstimator import RipperEstimator
from Tools.Estimators.RuleFitEstimator import RuleFitEstimator
from Tools.Estimators.XGBOOSTRegressor import XGBOOSTRegressor

def is_regression(model):
    return "Regressor" in str(model) or 'SVR' in str(model)


def is_regression_by_config(cfg):
    return cfg.get_params()['type_ml'].lower() == TypeML.REGRESSION.value


def is_regression_by_args(args):
    return args.regression


def is_penalty_weighted(args):
    return args['balanced'] != None and 'PEN' in args['balanced']


def is_tf_model(model):
    return 'tensorflow' in str(model)


def is_ripper_model(model):
    return 'RIPPER' in str(model)


def is_xgboost_model(model):
    return 'XGB' in str(model)


def is_rulefit_model(model):
    return 'RuleFit' in str(model)


def make_model(cfg, id_list, input_shape=None):
    return tf.keras.Sequential([
        #tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

def get_explainer_model(model, estimator, yts, cfg):
    if is_tf_model(model):
        return estimator
    elif is_ripper_model(model):
        return RipperEstimator(model, yts, cfg=cfg)
    elif is_rulefit_model(model):
        return RuleFitEstimator(model, yts, cfg=cfg)
    elif 'XGBRegressor' in str(model):
        return XGBOOSTRegressor(model)
    return model
