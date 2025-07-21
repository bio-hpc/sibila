#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from Tools.ToolsModels import is_regression_by_config
from Tools.Graphics import Graphics
import numpy as np
import re
from Tools.HTML.LIMEHTMLBuilder import LIMEHTMLBuilder
from lime import lime_tabular
from tqdm import tqdm
from pathlib import Path
from Common.Analysis.Explainers.ExplainerModel import ExplainerModel
from Tools.ToolsModels import is_rulefit_model
from Common.Config.ConfigHolder import ATTR, COLNAMES, FEATURE, STD, PROBA
import warnings
import sys

# Filter specific warnings for ill-conditioned matrices
warnings.filterwarnings('ignore', message='Ill-conditioned matrix.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*rcond.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=np.RankWarning)

class LimeExplainer(ExplainerModel):

    def explain(self):
        """
         https://github.com/marcotcr/lime
         https://lime-ml.readthedocs.io/en/latest/index.html
        """
        return self.execute()

    def execute(self):
        if not is_regression_by_config(self.cfg):
            explainer, predict_fn, n_samples = self.lime_classification()
        else:
            explainer, predict_fn, n_samples = self.lime_regression()

        # local interpretation
        prefix = Path(self.cfg.get_prefix()).stem
        self.html = LIMEHTMLBuilder()
        df_local = []
        colnames = [FEATURE, ATTR, 'range', 'class', PROBA]
        
        successful_explanations = 0
        failed_explanations = 0

        for i in tqdm(range(len(self.xts))):
            x = self.xts[i]
            
            # Try to explain instance with robust error handling
            exp = self._explain_instance_robust(explainer, x, predict_fn, n_samples, i)
            
            if exp is None:
                failed_explanations += 1
                continue
                
            successful_explanations += 1

            if is_regression_by_config(self.cfg):
                ypr = exp.predicted_value
                explanation = exp.as_list(0)
            else:
                ypr = exp.available_labels()[0]
                explanation = exp.as_list(ypr)

            self.html.append(exp.as_html(), sample_id=self.idx_xts[i])

            data = [[self.get_feature_name(e[0]), e[1], e[0], ypr, exp.local_pred[0]] for e in explanation]

            df = pd.DataFrame(data=data, columns=colnames)
            df_local.append(df)

            out_file = self.io_data.get_lime_folder() + "{}_Lime_explain_{}.csv".format(prefix, self.idx_xts[i])
            self.io_data.save_dataframe_cols(df, df.columns, out_file)
            del df

        self.html.close()
        
        print(f"LIME Explainer completed: {successful_explanations} success, {failed_explanations} failures")

        # Only proceed if we have at least some successful explanations
        if len(df_local) == 0:
            print("WARNING: Could not generate LIME explanations. Returning empty DataFrame.")
            return pd.DataFrame(columns=[FEATURE, ATTR, STD])

        # averaged attributions
        self.df_global = pd.concat(df_local)
        self.df_global = self.df_global[[FEATURE, ATTR]].groupby(FEATURE).agg(['mean','std']).reset_index()
        self.df_global.columns = [FEATURE, ATTR, STD]
        self.df_global = self.sort(self.df_global)
        return self.df_global

    def plot(self, df, method=None):
        # local explanations
        Graphics().plot_lime_html(self.html.get(), self.cfg.get_prefix() + '_Lime_tabular_explainer.html')

        # global explanation
        self.io_data.save_dataframe_cols(self.df_global, self.df_global.columns, self.cfg.get_prefix() + '_Lime.csv')
        aux_df = self.summarize(self.df_global)
        Graphics().plot_attributions(aux_df, 'LIME', self.cfg.get_prefix() + '_Lime.png', errors=self.get_errors(aux_df))

    def lime_classification(self):
        try:
            # Standard configuration
            explainer = lime_tabular.LimeTabularExplainer(self.xtr,
                                                          feature_names=self.id_list,
                                                          class_names=np.unique(self.yts, axis=0).astype(str),
                                                          discretize_continuous=True,
                                                          discretizer='entropy',
                                                          training_labels=self.ytr,
                                                          random_state=self.random_state,
                                                          feature_selection='forward_selection')
            return explainer, self.model.predict_proba, 5000
        except Exception as e:
            print(f"Failed standard LIME classification configuration: {e}")
            # More robust fallback configuration
            explainer = lime_tabular.LimeTabularExplainer(self.xtr,
                                                          feature_names=self.id_list,
                                                          class_names=np.unique(self.yts, axis=0).astype(str),
                                                          discretize_continuous=False,  # Less error-prone
                                                          training_labels=self.ytr,
                                                          random_state=self.random_state,
                                                          feature_selection='auto')  # Automatic selection
            return explainer, self.model.predict_proba, 1000  # Fewer samples

    def lime_regression(self):
        def lime_predict(x):
            try:
                pred = self.model.predict(x)
                return pred.reshape(pred.shape[0])
            except Exception as e:
                print(f"Error in LIME prediction: {e}")
                # Return default predictions in case of error
                return np.zeros(x.shape[0])

        try:
            # Standard configuration
            explainer = lime_tabular.LimeTabularExplainer(self.xtr,
                                                          feature_names=self.id_list,
                                                          discretize_continuous=True,
                                                          mode='regression')
            return explainer, lime_predict, 5000
        except Exception as e:
            print(f"Failed standard LIME regression configuration: {e}")
            # More robust fallback configuration
            explainer = lime_tabular.LimeTabularExplainer(self.xtr,
                                                          feature_names=self.id_list,
                                                          discretize_continuous=False,  # Less error-prone
                                                          mode='regression',
                                                          feature_selection='auto')
            return explainer, lime_predict, 1000  # Fewer samples

    def _explain_instance_robust(self, explainer, x, predict_fn, n_samples, instance_idx):
        """
        Attempts to explain an instance with fallback strategies to handle numerical errors
        """
        import warnings
        from numpy.linalg import LinAlgError
        
        # Strategy 1: Try with normal configuration
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return explainer.explain_instance(x, predict_fn, 
                                                num_features=len(self.id_list), 
                                                top_labels=1, 
                                                num_samples=n_samples)
        except (LinAlgError, ValueError, RuntimeError) as e:
            print(f"Failed on instance {instance_idx}: {str(e)[:100]}...")
        
        # Strategy 2: Reduce number of samples
        reduced_samples = [1000, 500, 100]
        for samples in reduced_samples:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return explainer.explain_instance(x, predict_fn,
                                                    num_features=len(self.id_list),
                                                    top_labels=1,
                                                    num_samples=samples)
            except (LinAlgError, ValueError, RuntimeError):
                continue
        
        # Strategy 3: Reduce number of features
        max_features = min(10, len(self.id_list))
        for num_feats in [max_features, 5, 3]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return explainer.explain_instance(x, predict_fn,
                                                    num_features=num_feats,
                                                    top_labels=1,
                                                    num_samples=100)
            except (LinAlgError, ValueError, RuntimeError):
                continue
        
        # Strategy 4: Create a new explainer with more robust configuration
        try:
            if not is_regression_by_config(self.cfg):
                robust_explainer = lime_tabular.LimeTabularExplainer(
                    self.xtr,
                    feature_names=self.id_list,
                    class_names=np.unique(self.yts, axis=0).astype(str),
                    discretize_continuous=False,  # Set to False to avoid problems
                    feature_selection='none',      # No feature selection
                    training_labels=self.ytr,
                    random_state=self.random_state
                )
            else:
                robust_explainer = lime_tabular.LimeTabularExplainer(
                    self.xtr,
                    feature_names=self.id_list,
                    discretize_continuous=False,
                    mode='regression',
                    feature_selection='none'
                )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return robust_explainer.explain_instance(x, predict_fn,
                                                       num_features=min(5, len(self.id_list)),
                                                       top_labels=1,
                                                       num_samples=50)
        except (LinAlgError, ValueError, RuntimeError):
            pass
        
        # If all strategies fail, return None
        print(f"Could not explain instance {instance_idx} with any strategy")
        return None

    def get_feature_name(self, e):
        m = re.split('[<]+ | [>]+ | [<=]+ | [>=]+ | [=]+', e)
        return m[1] if len(m) > 2 else m[0]

