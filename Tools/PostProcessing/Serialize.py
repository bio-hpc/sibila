#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Serialize.py:
"""
from Models import BaseModel


class Serialize:
    def __init__(self, model, xtr, ytr, xts, yts, id_list, cfg, io_data, idx_xts, run_method=None):
        """
        @param model:
        @param xtr:
        @param ytr:
        @param xts:
        @param yts:
        @param id_list:
        @param cfg:
        @param io_data:
        @param idx_xts:
        """
        self.file_model = BaseModel.get_filename_save_model(cfg, model)
        self.xtr = xtr
        self.ytr = ytr
        self.xts = xts
        self.yts = yts
        self.id_list = id_list
        self.cfg = cfg
        self.io_data = io_data
        self.idx_xts = idx_xts
        self.run_method = run_method  # only for a methode and called from a module

    def set_run_method(self, run_method):
        self.run_method = run_method

    def get_params(self):
        return dict({
            'model': BaseModel.load(self.file_model),
            'xtr': self.xtr,
            'ytr': self.ytr,
            'xts': self.xts,
            'yts': self.yts,
            'id_list': self.id_list,
            'cfg': self.cfg,
            'io_data': self.io_data,
            'idx_xts': self.idx_xts,
            'run_method': self.run_method
        })