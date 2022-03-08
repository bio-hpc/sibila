import json
from os.path import join
from Tools.IOData import IOData
from Common.Analysis.EvaluationMetrics import TypeML
from Tools.ToolsModels import is_regression_by_args

MAX_JSON_DEPTH = 0
PATH_CONFIG = join('Common', 'Config', 'DefaultConfigs')


def get_default_config(name_model, idata):
    file_json = join(PATH_CONFIG, '{}.json'.format(name_model))
    return idata.read_json(file_json)


def get_config(type_model, args):
    idata = IOData()
    default_cfg = get_default_config(type_model, idata)
    if args.parameters:
        for i in args.parameters:
            params = idata.read_json(i)
            if type_model.upper() == params['model'].upper(
            ):  # se ha introducido un fichero de configuracion para este modelo
                check_params(default_cfg, params)
    default_cfg['type_ml'] = TypeML.REGRESSION.value if is_regression_by_args(args) else TypeML.CLASSIFICATION.value
    return default_cfg


def get_basic_config(args):
    cfg = {'type_ml': TypeML.REGRESSION.value if is_regression_by_args(args) else TypeML.CLASSIFICATION.value}
    return cfg


def check_params(default_cfg, params, level=0):
    for k, v in params.items():
        if isinstance(v, dict):
            if level > MAX_JSON_DEPTH:
                default_cfg[k] = v
            else:
                check_params(default_cfg[k], params[k], level=level + 1)
        else:
            default_cfg[k] = v
