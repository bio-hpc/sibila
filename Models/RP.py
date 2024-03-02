from .BaseModel import BaseModel
from joblib import dump
from os.path import join
import wittgenstein as lw
from Tools.TypeML import TypeML
import numpy as np

PREFIX_OUT_DT = '{}_{}'


class RP(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(RP, self).__init__(io_data, cfg, id_list)
        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.model = lw.RIPPER(**self.cfg.get_params()['params'])
        else:
            print("Error: This model is only valid for classification")
            exit()

    def get_prefix(self):
        return join(self.cfg.get_folder(),
                    PREFIX_OUT_DT.format(self.cfg.get_params()['model'], self.cfg.get_name_dataset()))

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)

    def predict(self, xts):
        ypr = self.model_predict(xts)
        self.get_rules()
        return np.array(ypr).astype(int)

    def get_rules(self):
        f = open(self.cfg.get_prefix() + "_rules.txt", 'w')
        f.write('\n{}\n\n'.format(self.id_list))
        f.write('{}\n\n'.format(self.model.ruleset_.__str__()))
        for i in self.model.ruleset_:
            str_features = ""
            for k in i.__dict__['conds']:
                num_feature = k.__str__().split("=")[0]
                range_feature = k.__str__().split("=")[1]
                feature = self.id_list[int(num_feature)]
                str_features += feature + " " + range_feature + " and "

            str_features = str_features[:-4]
            f.write('{}\n'.format(str_features))
        f.write('\n')
        f.close()
