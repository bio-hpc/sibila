#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dice_ml
import pandas as pd
from dice_ml.utils import helpers
from tqdm import tqdm
from Tools.Graphics import Graphics
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Tools.ToolsModels import is_tf_model, is_ripper_model, is_rulefit_model, is_regression_by_config, is_multiclass
from Tools.Estimators.RipperEstimator import RipperEstimator
from pathlib import Path
from Common.Config.ConfigHolder import FEATURE, ATTR, STD, COLNAMES, PROBA, TRUEVAL, PREDVAL
from Tools.ToolsModels import is_regression_by_config

class DiceExplainer(ExplainerModel):

    DICE_METHOD = "random"  # random | genetic | kdtree

    def explain(self):
        if not is_ripper_model(self.model) and not is_rulefit_model(self.model):
            return self.execute_dice()

    def execute_dice(self):
        df = pd.DataFrame(self.xts, columns=self.id_list)
        df['class'] = self.yts
        lst_features = self.id_list

        d = dice_ml.Data(dataframe=df, continuous_features=lst_features, outcome_name='class')
        backend = 'TF2' if is_tf_model(self.model) else 'sklearn'
        feature_range = (self.xtr.min(axis=0) - 0.1, self.xtr.max(axis=0) + 0.1)

        model_type = 'classifier'
        desired_range = None
        sparsity = 'binary'
        if is_regression_by_config(self.cfg):
            model_type = 'regressor'
            desired_range = (min(self.ytr), max(self.ytr))
            sparsity = 'linear'

        m = dice_ml.Model(model=self.model, backend=backend, model_type=model_type)
        exp = dice_ml.Dice(d, m, method=self.DICE_METHOD)
        #exp.update_hyperparameters(feature_range=feature_range)    
        
        self.df_local = [None] * len(self.xts)
        desired_class = 0 if is_multiclass(self.cfg) else "opposite"

        try:
            query = pd.DataFrame(self.xts, columns=self.id_list)
            e1 = exp.generate_counterfactuals(
                    query,
                    total_CFs = 10,
                    desired_class = desired_class,
                    features_to_vary = lst_features,
                    posthoc_sparsity_algorithm = sparsity,
                    random_seed = self.cfg.get_args()['seed'],
                    desired_range = desired_range,
                    proximity_weight = 0.01,
                    diversity_weight = 0.01,
                    permitted_range = None
                    #feature_range = feature_range
                    #learning_rate = 0.1
            )

            if e1.cf_examples_list[0].final_cfs_df is not None:
                print('No counterfactuals were found')
                pass

            lst = [e1] if is_tf_model(self.model) else e1.cf_examples_list
            for i in tqdm(range(len(self.xts))):
                imp = exp.local_feature_importance(
                                  self.xts[i], 
                                  cf_examples_list = lst,
                                  desired_class = desired_class,
                                  desired_range = desired_range
                      )
                self.df_local[i] = pd.DataFrame(imp.local_importance).mean(axis=0).to_frame().reset_index()
                self.df_local[i].columns = [FEATURE, ATTR]
                if is_regression_by_config(self.cfg):
                    self.df_local[i][TRUEVAL] = self.yts[i]
                    self.df_local[i][PREDVAL] = self.proba_sample(self.xts[i])
                else:
                    self.df_local[i][PROBA] = self.proba_sample(self.xts[i])
        except BaseException as e:
            #print(str(e))
            pass
        finally:
            if 'e1' in locals():
                del e1
            if 'query' in locals():
                del query
            if 'imp' in locals():
                del imp

        self.df_local = [i for i in self.df_local if i is not None]

        if len(self.df_local) > 0:
            tmp = pd.concat(self.df_local)
            self.df_global = tmp.groupby(FEATURE)[ATTR].agg(['mean','std']).reset_index()
            self.df_global.columns = COLNAMES
            return self.df_global

        return None

    def plot(self, df, method=None):
        title = 'DiCE'

        # global interpretability
        Graphics().plot_attributions(df, title, self.cfg.get_prefix() + '_Dice.png', errors=self.get_errors(df))

        # local interpretability
        for i in tqdm(range(len(self.df_local))):
            filename = "{}_Dice_{}".format(Path(self.cfg.get_prefix()).stem, self.idx_xts[i])
            path_csv = "{}csv/{}.csv".format(self.io_data.get_dice_folder(), filename)
            path_png = "{}png/{}.png".format(self.io_data.get_dice_folder(), filename)

            # Sort in ascending order for plotting correctly
            df2 = self.df_local[i]

            self.io_data.save_dataframe_cols(df2, df2.columns, path_csv)

            # Add the real value into the label
            df2[FEATURE] = df2[FEATURE].apply(lambda x: '{:.3f}={}'.format(self.get_value(x, i), x))

            # Take the N most important features and sum up all the rest
            df2 = self.summarize(df2)

            Graphics().plot_attributions(df2, title, path_png, self.idx_xts[i])
            del df2

    def get_value(self, feature, row_id):
        index = self.id_list.index(feature)
        return self.xts[row_id, index]


