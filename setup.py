#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 10:36
# @Author  : Li Xiao
# @File    : setup.py
from setuptools import setup
from setuptools import find_packages

setup(
    name = 'MoGCN',
    version = '0.1',
    description = 'Multi-omics integration method using GCN and AE in Pytorch.',
    author = 'Li Xiao',
    author_email = 'lixiaoBioinfo@163.com',
    url = 'https://github.com/Lifoof',
    download_url = 'https://github.com/Lifoof',
    license = 'MIT',
    install_requires = ['numpy', 'pandas', 'scipy', 'torch'],
    packages = find_packages(),

    python_requires = '>=3.6'
)