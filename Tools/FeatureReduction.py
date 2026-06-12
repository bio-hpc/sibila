#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FeatureReduction.py:
    Statistical feature reduction pipeline for SIBILA datasets.
"""
import json
import numpy as np
import pandas as pd
from os.path import join, splitext, basename, isfile

from Tools.IOData import IOData

FIELD_TARGET = 'class'


class FeatureReduction:
    STATE_SUFFIX = '_feature_reduction.json'

    def __init__(self):
        self.io_data = IOData()

    @staticmethod
    def get_state_path(model_path):
        base, _ = splitext(model_path)
        return base + FeatureReduction.STATE_SUFFIX

    def save_backup(self, x, y, id_list, idx_samples, folder, dataset_path, io_data=None):
        io_data = io_data or self.io_data
        dataset_name = splitext(basename(dataset_path))[0]
        out_dir = join(folder, 'Dataset')
        io_data.create_dir_no_remove(out_dir)
        out_file = join(out_dir, '{}_original.csv'.format(dataset_name))

        df = pd.DataFrame(np.array(x, dtype=float), columns=id_list)
        df.insert(0, 'Sample ID', idx_samples)
        df[FIELD_TARGET] = y
        df.to_csv(out_file, index=False)
        io_data.print_m('Original dataset backup saved in {}'.format(out_file))

    def save_state(self, prefix, kept_features, removed_features, io_data=None):
        io_data = io_data or self.io_data
        state_file = prefix + self.STATE_SUFFIX
        state = {
            'kept_features': kept_features,
            'removed_features': removed_features
        }
        with open(state_file, 'w') as json_file:
            json.dump(state, json_file, indent=2)
        io_data.print_m('Feature reduction state saved in {}'.format(state_file))

    def load_state(self, model_path, io_data=None):
        io_data = io_data or self.io_data
        state_file = self.get_state_path(model_path)
        if not isfile(state_file):
            io_data.print_e('Feature reduction state not found: {}'.format(state_file))
        return io_data.read_json(state_file)

    def apply(self, x, y, id_list, fr_cfg, is_regression, io_data=None):
        io_data = io_data or self.io_data
        x = np.array(x, dtype=float)
        y = np.array(y)
        id_list = list(id_list)
        original_features = list(id_list)

        io_data.print_m('Feature reduction: {} features before processing'.format(len(id_list)))

        if fr_cfg.get('fill_nan') is not None:
            x = self._fill_nan(x, fr_cfg['fill_nan'], io_data)

        if fr_cfg.get('remove_empty_cols') is not None:
            x, id_list = self._remove_empty_cols(x, id_list, fr_cfg['remove_empty_cols'], io_data)

        if fr_cfg.get('drop_min_variance') is not None:
            x, id_list = self._drop_min_variance(x, id_list, fr_cfg['drop_min_variance'], io_data)

        if fr_cfg.get('outliers') is not None:
            x, y, id_list = self._remove_outliers(x, y, id_list, fr_cfg['outliers'], io_data)

        max_features = fr_cfg.get('max_features', 5)
        if fr_cfg.get('selection') is not None:
            x, id_list = self._select_features(
                x, y, id_list, fr_cfg['selection'], max_features, is_regression, io_data
            )

        kept_features = list(id_list)
        removed_features = [feature for feature in original_features if feature not in kept_features]
        io_data.print_m('Feature reduction: {} features after processing'.format(len(kept_features)))

        state = {
            'kept_features': kept_features,
            'removed_features': removed_features
        }
        return x, id_list, y, state

    def apply_from_state(self, x, id_list, model_path, io_data=None):
        io_data = io_data or self.io_data
        state = self.load_state(model_path, io_data)
        kept_features = state['kept_features']
        id_list = list(id_list)

        missing = [feature for feature in kept_features if feature not in id_list]
        if missing:
            io_data.print_e('Prediction dataset is missing features required by the model: {}'.format(', '.join(missing)))

        indices = [id_list.index(feature) for feature in kept_features]
        x = np.array(x, dtype=float)[:, indices]
        io_data.print_m('Feature reduction (prediction): using {} features'.format(len(kept_features)))
        return x, kept_features

    @staticmethod
    def _fill_nan(x, strategy, io_data):
        io_data.print_m('\tApplying fill_nan: {}'.format(strategy))
        x = x.copy()
        if isinstance(strategy, (int, float)):
            fill_value = float(strategy)
        elif str(strategy).lower() == 'mean':
            fill_value = None
        elif str(strategy).lower() == 'zero':
            fill_value = 0.0
        else:
            try:
                fill_value = float(strategy)
            except (TypeError, ValueError):
                io_data.print_e('Unsupported fill_nan strategy: {}'.format(strategy))

        if fill_value is None:
            col_means = np.zeros(x.shape[1], dtype=float)
            for idx in range(x.shape[1]):
                valid = x[:, idx][~np.isnan(x[:, idx])]
                if valid.size:
                    col_means[idx] = valid.mean()
            nan_mask = np.isnan(x)
            x[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        else:
            x[np.isnan(x)] = fill_value
        return x

    @staticmethod
    def _remove_empty_cols(x, id_list, threshold, io_data):
        io_data.print_m('\tApplying remove_empty_cols: {}'.format(threshold))
        keep_indices = []
        keep_names = []
        for idx, name in enumerate(id_list):
            column = x[:, idx]
            missing_ratio = np.mean(np.isnan(column) | (column == 0))
            if missing_ratio < threshold:
                keep_indices.append(idx)
                keep_names.append(name)
        if not keep_indices:
            io_data.print_e('remove_empty_cols removed all features')
        return x[:, keep_indices], keep_names

    @staticmethod
    def _drop_min_variance(x, id_list, threshold, io_data):
        io_data.print_m('\tApplying drop_min_variance: {}'.format(threshold))
        variances = np.var(x, axis=0)
        support = variances > float(threshold)
        x_reduced = x[:, support]
        id_list = [name for name, keep in zip(id_list, support) if keep]
        if x_reduced.shape[1] == 0:
            io_data.print_e('drop_min_variance removed all features')
        return x_reduced, id_list

    @staticmethod
    def _remove_outliers(x, y, id_list, method, io_data):
        io_data.print_m('\tApplying outliers: {}'.format(method))
        method = str(method).lower()
        keep_mask = np.ones(x.shape[0], dtype=bool)

        if method == 'turkey':
            for idx in range(x.shape[1]):
                column = x[:, idx]
                q1, q3 = np.percentile(column, [25, 75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                keep_mask &= (column >= lower) & (column <= upper)
        elif method == 'remove':
            for idx in range(x.shape[1]):
                column = x[:, idx]
                std = np.std(column)
                if std == 0:
                    continue
                z_scores = np.abs((column - np.mean(column)) / std)
                keep_mask &= z_scores <= 3
        else:
            io_data.print_e('Unsupported outliers method: {}'.format(method))

        removed_rows = x.shape[0] - np.sum(keep_mask)
        io_data.print_m('\tRemoved {} outlier rows'.format(removed_rows))
        return x[keep_mask], y[keep_mask], id_list

    @staticmethod
    def _f_classif(x, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples = x.shape[0]
        scores = np.zeros(x.shape[1], dtype=float)

        for idx in range(x.shape[1]):
            column = x[:, idx]
            overall_mean = column.mean()
            ss_between = 0.0
            ss_within = 0.0
            for label in classes:
                group = column[y == label]
                group_mean = group.mean()
                ss_between += len(group) * (group_mean - overall_mean) ** 2
                ss_within += ((group - group_mean) ** 2).sum()

            df_between = n_classes - 1
            df_within = n_samples - n_classes
            if ss_within == 0 or df_between <= 0 or df_within <= 0:
                scores[idx] = 0.0
            else:
                scores[idx] = (ss_between / df_between) / (ss_within / df_within)
        return scores

    @staticmethod
    def _f_regression(x, y):
        y = y.astype(float)
        y_centered = y - y.mean()
        y_ss = np.dot(y_centered, y_centered)
        scores = np.zeros(x.shape[1], dtype=float)

        for idx in range(x.shape[1]):
            column = x[:, idx].astype(float)
            column_centered = column - column.mean()
            column_ss = np.dot(column_centered, column_centered)
            if column_ss == 0 or y_ss == 0:
                scores[idx] = 0.0
            else:
                corr = np.dot(column_centered, y_centered) / np.sqrt(column_ss * y_ss)
                scores[idx] = corr ** 2
        return scores

    @classmethod
    def _score_features(cls, x, y, is_regression):
        if is_regression:
            scores = cls._f_regression(x, y)
        else:
            scores = cls._f_classif(x, y)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_features(self, x, y, id_list, method, max_features, is_regression, io_data):
        method = str(method).lower()
        io_data.print_m('\tApplying selection: {} (max_features={})'.format(method, max_features))
        max_features = int(max_features)
        n_features = x.shape[1]

        if n_features <= max_features:
            io_data.print_m('\tSelection skipped: already at or below max_features')
            return x, id_list

        if method == 'forward':
            scores = self._score_features(x, y, is_regression)
            ranked = np.argsort(scores)[::-1]
            selected = sorted(ranked[:max_features].tolist())
        elif method == 'backward':
            selected = list(range(n_features))
            while len(selected) > max_features:
                scores = self._score_features(x[:, selected], y, is_regression)
                worst_local = int(np.argmin(scores))
                selected.pop(worst_local)
            selected = sorted(selected)
        else:
            io_data.print_e('Unsupported selection method: {}'.format(method))

        selected = sorted(selected)
        return x[:, selected], [id_list[idx] for idx in selected]
