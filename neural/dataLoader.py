#!/usr/bin/env python
import _pickle as cPickle
import gzip
import numpy as np 


def load_data():
    f = gzip.open('./dataSource/mnist.pkl.gz', 'rb')
    trainning_data, validation_data, test_data = cPickle.load(f)
    f.close()

    return (trainning_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    trainning_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    trainning_results = [vectorized_result(y) for y in tr_d[1]]
    trainning_data = zip(trainning_inputs, trainning_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (trainning_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

