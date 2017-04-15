#!/usr/bin/env python
# -*-coding:UTF-8 -*-

import neuralTrain
import dataLoader
import img2mat


def runNeural():
    training_data, validation_data, test_data = dataLoader.load_data_wrapper()
    net = neuralTrain.Network([784, 30, 10])
    net.SGD(training_data, 3, 10, 3.0, test_data=test_data)

    for i in range(0, 10):
        grayData = img2mat.rgb2gray("./dataNum/{}A.png".format(i))
        print("input {}A output {}".format(i, net.testOutput(grayData)))

        grayData = img2mat.rgb2gray("./dataNum/{}.png".format(i))
        print("input {} output {}".format(i, net.testOutput(grayData)))


if __name__ == '__main__':
    runNeural()

