#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mnist_loader.py
# @Author: hrt
# @Date  : 18-10-7
# @Desc  :

import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_data, valid_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return train_data, valid_data, test_data

def load_data_wrapper():
    train_data, valid_data, test_data = load_data()

    train_X, train_y = train_data
    train_X = train_X.reshape((-1, 784, 1))
    train_data = zip(train_X, train_y)

    valid_X, valid_y = valid_data
    valid_X = valid_X.reshape((-1, 784, 1))
    valid_data = zip(valid_X, valid_y)

    test_X, test_y = test_data
    test_X = test_X.reshape((-1, 784, 1))
    test_data = zip(test_X, test_y)

    return train_data, valid_data, test_data
