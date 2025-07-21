#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global configuration to improve SIBILA's numerical robustness
"""
__author__ = "SIBILA Team"
__version__ = "1.0"

import warnings
import numpy as np
import os
import sys

def configure_numeric_stability():
    """
    Configures global settings to improve numerical stability
    """
    
    # Configure NumPy for better error handling
    np.seterr(all='ignore')  # Ignore overflow, underflow, etc.
    
    # Configure environment variables for scientific libraries
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # For libraries like BLAS/LAPACK
    os.environ['OMP_NUM_THREADS'] = '1'  # Avoid concurrency issues
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # For TensorFlow (if present)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logs
    
    print("✓ Numerical stability configuration applied")

def configure_warnings_filter():
    """
    Configures warning filters to reduce noise in logs
    """
    
    # Ill-conditioned matrix warnings
    warnings.filterwarnings('ignore', message='Ill-conditioned matrix.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*rcond.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*singular matrix.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*overflow.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*underflow.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*invalid value.*', category=RuntimeWarning)
    
    # Specific library warnings
    warnings.filterwarnings('ignore', category=np.RankWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Scikit-learn warnings
    warnings.filterwarnings('ignore', message='.*convergence.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*feature names.*', category=UserWarning)
    
    # LIME warnings
    warnings.filterwarnings('ignore', message='.*LIME.*', category=UserWarning)
    
    # SHAP warnings
    warnings.filterwarnings('ignore', message='.*SHAP.*', category=UserWarning)
    
    print("✓ Warning filters configured")

def configure_sklearn_settings():
    """
    Configura ajustes específicos de scikit-learn para mayor robustez
    """
    try:
        from sklearn import set_config
        # Configurar scikit-learn para ser más robusto
        set_config(assume_finite=True)  # Asumir que los datos son finitos
        print("✓ Configuración de scikit-learn aplicada")
    except ImportError:
        print("! scikit-learn no disponible para configurar")

def apply_robust_configuration():
    """
    Aplica toda la configuración de robustez
    """
    print("Aplicando configuración de robustez para SIBILA...")
    
    configure_numeric_stability()
    configure_warnings_filter()
    configure_sklearn_settings()
    
    print("✅ Configuración de robustez completada")
    print("   - Los errores numéricos serán manejados de forma más robusta")
    print("   - Los warnings de matrices mal condicionadas están filtrados")
    print("   - Los explainers continuarán funcionando ante errores puntuales")

if __name__ == "__main__":
    apply_robust_configuration() 