#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixin class to add robustness to interpretability explainers
"""
__author__ = "SIBILA Team"
__version__ = "1.0"

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Any, Callable
import logging

class RobustExplainerMixin:
    """
    Mixin that provides methods for robust error handling in explainers
    """
    
    @staticmethod
    def setup_warnings_filter():
        """Configures warning filters for explainers"""
        warnings.filterwarnings('ignore', message='Ill-conditioned matrix.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*rcond.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*singular matrix.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*overflow.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=np.RankWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def safe_explain_wrapper(self, explain_func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        Safe wrapper for explanation functions
        
        Args:
            explain_func: Explanation function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Explanation result or None if it fails
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return explain_func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Error in explanation: {str(e)[:100]}...")
            return None
    
    def create_fallback_dataframe(self, feature_names: list, default_value: float = 0.0) -> pd.DataFrame:
        """
        Creates a fallback DataFrame when explanation fails
        
        Args:
            feature_names: List of feature names
            default_value: Default value for attributions
            
        Returns:
            DataFrame with default values
        """
        from Common.Config.ConfigHolder import FEATURE, ATTR, STD
        
        n_features = len(feature_names)
        return pd.DataFrame({
            FEATURE: feature_names,
            ATTR: [default_value] * n_features,
            STD: [0.1] * n_features  # Small standard deviation
        })
    
    def reduce_sample_size(self, data: np.ndarray, max_size: int = 100) -> np.ndarray:
        """
        Reduces sample size to avoid computational problems
        
        Args:
            data: Original data
            max_size: Maximum desired size
            
        Returns:
            Reduced data
        """
        if len(data) <= max_size:
            return data
        
        # Stratified random selection if possible
        indices = np.random.choice(len(data), size=max_size, replace=False)
        return data[indices]
    
    def log_explainer_stats(self, successful: int, failed: int, method_name: str):
        """
        Logs explainer success/failure statistics
        
        Args:
            successful: Number of successful explanations
            failed: Number of failed explanations
            method_name: Name of the explanation method
        """
        total = successful + failed
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"{method_name} completed:")
        print(f"  - Successful: {successful}/{total} ({success_rate:.1f}%)")
        print(f"  - Failed: {failed}/{total} ({100-success_rate:.1f}%)")
        
        if failed > 0:
            print(f"  - WARNING: {failed} explanations failed") 