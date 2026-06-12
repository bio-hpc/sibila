#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic synthetic datasets for SIBILA unit tests."""
import numpy as np


CLASSIFICATION_FEATURE_NAMES = ['height_cm', 'age', 'num_tatoos', 'piercings']
REGRESSION_FEATURE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']


def _rng(seed):
    return np.random.RandomState(seed)


def make_classification_dataset(n_samples=100, n_features=4, seed=2021, imbalanced=False):
    """
    Build a binary classification dataset compatible with SIBILA tests.
    """
    random = _rng(seed)
    feature_names = CLASSIFICATION_FEATURE_NAMES[:n_features]
    if len(feature_names) < n_features:
        feature_names.extend(['feature_{}'.format(i) for i in range(len(feature_names), n_features)])

    idx_samples = list(range(n_samples))
    x = np.zeros((n_samples, n_features), dtype=float)
    x[:, 0] = random.normal(175, 15, n_samples)
    if n_features > 1:
        x[:, 1] = random.normal(35, 12, n_samples)
    if n_features > 2:
        x[:, 2] = random.poisson(12, n_samples)
    if n_features > 3:
        x[:, 3] = random.poisson(3, n_samples)
    for idx in range(4, n_features):
        x[:, idx] = random.normal(idx + 1, 1.5, n_samples)

    weights = np.array([0.03, 0.04, 0.08, 0.15][:n_features], dtype=float)
    if n_features > len(weights):
        extra = np.linspace(0.2, 0.5, n_features - len(weights))
        weights = np.concatenate([weights, extra])

    scores = x.dot(weights) + random.normal(0, 0.2, n_samples)
    y = (scores >= np.median(scores)).astype(int)

    if imbalanced:
        majority_class = int(np.bincount(y).argmax())
        minority_class = 1 - majority_class
        minority_size = max(5, n_samples // 10)
        majority_size = n_samples - minority_size
        minority_idx = random.choice(n_samples, minority_size, replace=False)
        y[:] = majority_class
        y[minority_idx] = minority_class
        order = random.permutation(n_samples)
        x = x[order]
        y = y[order]
        idx_samples = [idx_samples[i] for i in order]

    target_classes = len(np.unique(y))
    return x, y, feature_names, idx_samples, target_classes


def make_regression_dataset(n_samples=71, n_features=6, seed=2021):
    """
    Build a regression dataset compatible with SIBILA tests.
    """
    random = _rng(seed)
    feature_names = REGRESSION_FEATURE_NAMES[:n_features]
    if len(feature_names) < n_features:
        feature_names.extend(['feature_{}'.format(i) for i in range(len(feature_names), n_features)])

    idx_samples = list(range(n_samples))
    x = random.rand(n_samples, n_features)
    coefficients = np.linspace(50, 120, n_features)
    y = x.dot(coefficients) + random.normal(0, 5, n_samples)
    return x, y, feature_names, idx_samples, None
