#!/usr/bin/env python
# -*-coding:UTF-8 -*-

import neuralTrain
import dataLoader

def runNeural():
    training_data, validation_data, test_data = dataLoader.load_data_wrapper()
    net = neuralTrain.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == '__main__':
    runNeural()