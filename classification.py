#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : classification.py
# @Author: hrt
# @Date  : 18-10-7
# @Desc  :

import random
import numpy as np
import matplotlib.pyplot as plt

from opr import sigmoid, relu, one_hot
from mnist_loader import load_data_wrapper

class Network():
    def __init__(self, node_list):
        self.num_layers = len(node_list)
        self.node_list = node_list

        self.weight = [np.random.randn(node_after, node_before) for node_before, node_after in zip(node_list[:-1], node_list[1:])]
        self.bias = [np.random.randn(node, 1) for node in node_list[1:]]

        self.init_data()

    def forward(self, x):
        for w, b in zip(self.weight, self.bias):
            x = sigmoid(np.dot(w, x) + b)

        return x

    def backword(self, x, y):
        delta_w = [np.zeros(w.shape) for w in self.weight]
        delta_b = [np.zeros(b.shape) for b in self.bias]

        activations = [x]
        for w,b in zip(self.weight, self.bias):
            x = sigmoid(np.dot(w, x) + b)
            activations.append(x)

        delta = (activations[-1] - y) * (activations[-1] * (1 - activations[-1]))

        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            delta = np.dot(self.weight[-i+1].transpose(), delta) * (activations[-i] * (1 - activations[-i]))
            delta_b[-i] = delta
            delta_w[-i] = np.dot(delta, activations[-i-1].transpose())

        return (delta_w, delta_b)

    def train(self, nr_epoch=10, batch_size=32, lr=0.01, show_plot=True):
        train_errors = []
        test_accs = []

        for i in range(nr_epoch):
            for batch_data in self.get_batch(batch_size):
                delta_w = [np.zeros(w.shape) for w in self.weight]
                delta_b = [np.zeros(b.shape) for b in self.bias]

                for x, y in batch_data:
                    one_hot_y = one_hot(y)

                    gradient_w, gradient_b = self.backword(x, one_hot_y)

                    delta_w = [a+b for a,b in zip(delta_w, gradient_w)]
                    delta_b = [a+b for a,b in zip(delta_b, gradient_b)]

                # update weights and bias
                self.weight = [w - lr * dw for w, dw in zip(self.weight, delta_w)]
                self.bias = [b - lr * db for b, db in zip(self.bias, delta_b)]

            # train error
            train_error = 1.0 - float(self.eval(self.train_data)) / self.num_train
            train_errors.append(train_error)

            # test accuracy
            test_acc = float(self.eval(self.test_data)) / self.num_test
            test_accs.append(test_acc)

            print("Epoch {}: train error {:.2f}%, test accuracy {:.2f}%".format(i + 1, train_error*100, test_acc*100))

        if show_plot:
            self.plot(train_errors, test_accs)

    def plot(self, train_errors, test_accs):
        plt.plot(train_errors)
        plt.title('train error')
        plt.show()

        plt.plot(test_accs)
        plt.title('test accuracy')
        plt.show()

    def init_data(self):
        # load_data
        train_data, valid_data, test_data = load_data_wrapper()
        self.train_data = list(train_data)
        self.valid_data = list(valid_data)
        self.test_data = list(test_data)

        # param
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        self.num_test = len(self.test_data)

    def get_batch(self, batch_size):
        np.random.shuffle(self.train_data)

        i = 0

        while i < self.num_train:
            yield self.train_data[i:i+batch_size]

            i += batch_size

    def eval(self, data):
        result = [(np.argmax(self.forward(x)), y) for (x,y) in data]

        return sum(int(x == y) for (x,y) in result)

if __name__ == '__main__':
    network = Network([784, 128, 64, 32, 16, 10])

    network.train(128, 32, 0.1)