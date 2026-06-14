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
        n_samples_initial = x.shape[0]

        io_data.print_m(
            'Feature reduction: {} samples x {} features before processing'.format(
                n_samples_initial, len(id_list)
            )
        )

        if fr_cfg.get('fill_nan') is not None:
            x = self._fill_nan(x, fr_cfg['fill_nan'], io_data)

        if fr_cfg.get('remove_empty_cols') is not None:
            x, id_list = self._remove_empty_cols(x, id_list, fr_cfg['remove_empty_cols'], io_data)

        if fr_cfg.get('drop_min_variance') is not None:
            x, id_list = self._drop_min_variance(x, id_list, fr_cfg['drop_min_variance'], io_data)

        if fr_cfg.get('outliers') is not None:
            x = self._cap_outliers(x, fr_cfg['outliers'], io_data)

        if fr_cfg.get('selection') is not None:
            x, id_list = self._select_features(
                x, y, id_list, fr_cfg, is_regression, io_data
            )

        kept_features = list(id_list)
        removed_features = [feature for feature in original_features if feature not in kept_features]
        io_data.print_m(
            'Feature reduction finished: final dataset {} samples x {} features '
            '(removed {} features)'.format(
                x.shape[0],
                len(kept_features),
                len(removed_features),
            )
        )

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
        io_data.print_m(
            'Feature reduction (prediction): final dataset {} samples x {} features'.format(
                x.shape[0], len(kept_features)
            )
        )
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
    def _cap_outliers(x, method, io_data):
        io_data.print_m('\tApplying outliers: {}'.format(method))
        method = str(method).lower()
        x = x.copy()
        capped_values = 0

        if method == 'turkey':
            for idx in range(x.shape[1]):
                column = x[:, idx]
                q1, q3 = np.percentile(column, [25, 75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                below = column < lower
                above = column > upper
                capped_values += int(below.sum() + above.sum())
                column = column.copy()
                column[below] = lower
                column[above] = upper
                x[:, idx] = column
        elif method == 'remove':
            for idx in range(x.shape[1]):
                column = x[:, idx]
                std = np.std(column)
                if std == 0:
                    continue
                mean = np.mean(column)
                lower = mean - 3 * std
                upper = mean + 3 * std
                below = column < lower
                above = column > upper
                capped_values += int(below.sum() + above.sum())
                column = column.copy()
                column[below] = lower
                column[above] = upper
                x[:, idx] = column
        else:
            io_data.print_e('Unsupported outliers method: {}'.format(method))

        io_data.print_m('\tCapped {} outlier values (all rows preserved)'.format(capped_values))
        return x

    @staticmethod
    def _f_distribution_pvalue(f_score, df_num, df_den):
        if f_score <= 0 or df_num <= 0 or df_den <= 0:
            return 1.0
        try:
            from scipy.stats import f as f_dist
            return float(f_dist.sf(f_score, df_num, df_den))
        except Exception:
            # Chi-square normal approximation when scipy is unavailable.
            x = df_num * f_score
            mu = df_num
            sigma = max(np.sqrt(2.0 * df_num), 1e-12)
            z = (x - mu) / sigma
            from math import erfc, sqrt
            return float(0.5 * erfc(z / sqrt(2.0)))

    @classmethod
    def _f_classif(cls, x, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples = x.shape[0]
        scores = np.zeros(x.shape[1], dtype=float)
        p_values = np.ones(x.shape[1], dtype=float)
        df_between = n_classes - 1
        df_within = n_samples - n_classes

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

            if ss_within == 0 or df_between <= 0 or df_within <= 0:
                scores[idx] = 0.0
                p_values[idx] = 1.0
            else:
                f_score = (ss_between / df_between) / (ss_within / df_within)
                scores[idx] = f_score
                p_values[idx] = cls._f_distribution_pvalue(f_score, df_between, df_within)
        return scores, p_values

    @classmethod
    def _f_regression(cls, x, y):
        y = y.astype(float)
        y_centered = y - y.mean()
        y_ss = np.dot(y_centered, y_centered)
        n_samples = x.shape[0]
        scores = np.zeros(x.shape[1], dtype=float)
        p_values = np.ones(x.shape[1], dtype=float)
        df_num = 1
        df_den = n_samples - 2

        for idx in range(x.shape[1]):
            column = x[:, idx].astype(float)
            column_centered = column - column.mean()
            column_ss = np.dot(column_centered, column_centered)
            if column_ss == 0 or y_ss == 0 or df_den <= 0:
                scores[idx] = 0.0
                p_values[idx] = 1.0
            else:
                corr = np.dot(column_centered, y_centered) / np.sqrt(column_ss * y_ss)
                r2 = corr ** 2
                f_score = (r2 / max(1.0 - r2, 1e-12)) * df_den
                scores[idx] = f_score
                p_values[idx] = cls._f_distribution_pvalue(f_score, df_num, df_den)
        return scores, p_values

    @classmethod
    def _score_features(cls, x, y, is_regression):
        if is_regression:
            scores, p_values = cls._f_regression(x, y)
        else:
            scores, p_values = cls._f_classif(x, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        return scores, p_values

    @staticmethod
    def _parse_selection_config(fr_cfg):
        max_features = fr_cfg.get('max_features')
        if max_features is not None:
            max_features = int(max_features)

        min_features = fr_cfg.get('min_features', 1)
        min_features = max(1, int(min_features))

        p_value_threshold = fr_cfg.get('p_value_threshold')
        if p_value_threshold is not None:
            p_value_threshold = float(p_value_threshold)

        score_threshold = fr_cfg.get('score_threshold')
        if score_threshold is not None:
            score_threshold = float(score_threshold)

        return {
            'method': str(fr_cfg['selection']).lower(),
            'max_features': max_features,
            'min_features': min_features,
            'p_value_threshold': p_value_threshold,
            'score_threshold': score_threshold,
        }

    @staticmethod
    def _is_significant(p_value, score, selection_cfg):
        if selection_cfg['p_value_threshold'] is not None:
            return p_value <= selection_cfg['p_value_threshold']
        if selection_cfg['score_threshold'] is not None:
            return score >= selection_cfg['score_threshold']
        return True

    def _apply_selection_limits(self, selected, scores, p_values, selection_cfg, io_data):
        selected = sorted(set(selected))
        min_features = selection_cfg['min_features']
        max_features = selection_cfg['max_features']

        if selection_cfg['p_value_threshold'] is not None or selection_cfg['score_threshold'] is not None:
            significant = [
                idx for idx in selected
                if self._is_significant(p_values[idx], scores[idx], selection_cfg)
            ]
            if len(significant) >= min_features:
                selected = significant
            else:
                ranked = sorted(selected, key=lambda idx: (p_values[idx], -scores[idx]))
                selected = ranked[:min_features]
                io_data.print_m('\tSelection kept {} features to satisfy min_features'.format(len(selected)))

        if max_features is not None and len(selected) > max_features:
            ranked = sorted(selected, key=lambda idx: (-scores[idx], p_values[idx]))
            selected = sorted(ranked[:max_features])
            io_data.print_m('\tSelection capped at max_features={}'.format(max_features))

        return selected

    def _select_features(self, x, y, id_list, fr_cfg, is_regression, io_data):
        selection_cfg = self._parse_selection_config(fr_cfg)
        method = selection_cfg['method']
        max_features = selection_cfg['max_features']
        p_value_threshold = selection_cfg['p_value_threshold']
        score_threshold = selection_cfg['score_threshold']

        if max_features is None and p_value_threshold is None and score_threshold is None:
            io_data.print_e(
                'Feature selection requires at least one stopping criterion: '
                'p_value_threshold, score_threshold or max_features'
            )

        stop_desc = []
        if p_value_threshold is not None:
            stop_desc.append('p_value_threshold={}'.format(p_value_threshold))
        if score_threshold is not None:
            stop_desc.append('score_threshold={}'.format(score_threshold))
        if max_features is not None:
            stop_desc.append('max_features={}'.format(max_features))
        io_data.print_m('\tApplying selection: {} ({})'.format(method, ', '.join(stop_desc)))

        n_features = x.shape[1]
        scores, p_values = self._score_features(x, y, is_regression)

        if method == 'forward':
            ranked = sorted(range(n_features), key=lambda idx: (-scores[idx], p_values[idx]))
            selected = []
            for idx in ranked:
                if not self._is_significant(p_values[idx], scores[idx], selection_cfg):
                    break
                selected.append(idx)
                if max_features is not None and len(selected) >= max_features:
                    break
            if len(selected) < selection_cfg['min_features']:
                selected = ranked[:selection_cfg['min_features']]
        elif method == 'backward':
            selected = list(range(n_features))
            while len(selected) > selection_cfg['min_features']:
                if max_features is not None and len(selected) <= max_features:
                    break

                local_scores, local_p_values = self._score_features(x[:, selected], y, is_regression)
                removable = [
                    pos for pos, idx in enumerate(selected)
                    if not self._is_significant(local_p_values[pos], local_scores[pos], selection_cfg)
                ]
                if not removable:
                    if max_features is None or len(selected) <= max_features:
                        break
                    worst_local = int(np.argmin(local_scores))
                    selected.pop(worst_local)
                    continue

                worst_removable = min(
                    removable,
                    key=lambda pos: (local_scores[pos], -local_p_values[pos])
                )
                selected.pop(worst_removable)
        else:
            io_data.print_e('Unsupported selection method: {}'.format(method))

        selected = self._apply_selection_limits(selected, scores, p_values, selection_cfg, io_data)
        return x[:, selected], [id_list[idx] for idx in selected]
