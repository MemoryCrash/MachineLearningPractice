#!/usr/bin/env python
# -*-coding:UTF-8 -*-

import pickle
import gzip
import numpy as np


def load_data():

    with gzip.open('./dataSource/mnist.pkl.gz', 'rb') as f:
        trainning_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    return (trainning_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    trainning_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    trainning_results = [vectorized_result(y) for y in tr_d[1]]
    trainning_data = list(zip(trainning_inputs, trainning_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (trainning_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == '__main__':

    load_data_wrapper()
