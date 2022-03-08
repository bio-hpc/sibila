#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    LIMEHTMLBuilder.py:
    Build a single HTML file with all the LIME plots.
"""
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import bs4
import html

class LIMEHTMLBuilder():

    def __init__(self):
        self.html = '<html>\n<meta http-equiv="content-type" content="text/html; charset=utf-8">\n'
        self.header_exists = False
        self.body_exists = False
        self.parser = None

    def append(self, html, sample_id=None):
        self.parser = bs4.BeautifulSoup(html, 'html.parser')
        if not self.header_exists:
            self.add_code('<head>' + self.get_content('head') + '</head>')
            self.header_exists = True

        if not self.body_exists:
            self.html += '<body>'
            self.body_exists = True

        if sample_id is not None:
            self.add_code('Sample #{}'.format(sample_id))

        self.add_code(self.get_content('body'))

    def add_code(self, html):
        self.html += '\n' + html

    def close(self):
        if self.body_exists:
            self.html += '\n</body>'
        self.html += '\n</html>'

    def get(self):
        return self.html

    def get_content(self, tag):
        return '\n'.join(list(map(str, self.parser.find(tag).contents)))
