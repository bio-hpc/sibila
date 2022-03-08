#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ClassFactory.py:
    Creates a new instance of an object from its classname and arguments

"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"


class ClassFactory():

	def __init__(self, classname, **kwargs):
		self.__import_module(classname)
		self.obj = self.obj(**kwargs)

	""" Returns the newly created object """
	def get(self):
		return self.obj

	""" Imports all modules in the path """
	def __import_module(self, classname):
		components = classname.split('.')
		self.obj = __import__(components[0])
		for comp in components[1:]:
			self.obj = getattr(self.obj, comp)
