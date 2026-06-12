#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Default interpretability parameters for SIBILA explainers."""

DEFAULT_INTERPRETABILITY = {
    "Lime": {
        "num_samples": 5000,
        "discretize_continuous": True,
        "discretizer": "entropy",
        "feature_selection": "forward_selection",
        "top_labels": 1
    },
    "Shapley": {},
    "IntegratedGradients": {
        "method": "riemann_trapezoid",
        "n_steps": 50,
        "baseline": "mean",
        "surrogate": {
            "hidden_units": [64, 32],
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "verbose": 0
        }
    },
    "PermutationImportance": {
        "n_repeats": 5,
        "n_jobs": 1
    },
    "MDI": {
        "n_estimators": 100,
        "random_state": None
    },
    "Counterfactuals": {
        "method": "random",
        "total_CFs": 10,
        "posthoc_sparsity_algorithm": "binary"
    },
    "Anchor": {
        "threshold": 0.95,
        "disc_perc": [25, 50, 75]
    },
    "PDP": {},
    "ALE": {}
}


def get_explainer_config(model_params, explainer_name):
    """
    Merge user interpretability settings from model JSON with defaults.
    """
    defaults = DEFAULT_INTERPRETABILITY.get(explainer_name, {}).copy()
    user_cfg = model_params.get('interpretability', {}).get(explainer_name, {})

    merged = defaults.copy()
    for key, value in user_cfg.items():
        if key == 'surrogate' and isinstance(value, dict):
            surrogate = defaults.get('surrogate', {}).copy()
            surrogate.update(value)
            merged['surrogate'] = surrogate
        else:
            merged[key] = value
    return merged
