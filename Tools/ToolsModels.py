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
    #if input_shape == None:
    #    input_shape = (1, len(id_list))

    #layers = [tf.keras.layers.InputLayer(input_shape=input_shape)]
    #for k in cfg.get_params()['params']['layers'].keys():
    #    layer_conf = cfg.get_params()['params']['layers'][k]
    #    if not 'properties' in layer_conf.keys():
    #        layer_conf['properties'] = {}

    #    # https://keras.io/api/layers/initializers/
    #    layer = ClassFactory(layer_conf['type'], **layer_conf['properties']).get()
    #    try:
    #        if 'kernel_initializer' in dir(layer):
    #            layer.kernel_initializer = tf.keras.initializers.GlorotUniform(seed=cfg.get_args()['seed'])
    #    except:
    #        layer._kernel_initializer = tf.keras.initializers.GlorotUniform(seed=cfg.get_args()['seed'])

    #    layers.append(layer)

    #return tf.keras.Sequential(layers)
 
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
