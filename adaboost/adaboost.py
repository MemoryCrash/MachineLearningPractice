#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from numpy import *

def loadSimpData():
    datMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.],]
        )
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datMat, classLabels


if __name__ == '__main__':
    datMat, classLabels = loadSimpData()

    print(datMat)