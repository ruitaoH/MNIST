#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : opr.py
# @Author: hrt
# @Date  : 18-10-7
# @Desc  :

import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def one_hot(i):
    v = np.zeros((10 ,1))
    v[i] = 1.0

    return v