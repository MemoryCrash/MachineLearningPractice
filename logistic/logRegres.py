#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from numpy import *
import operator


def loadDataSet():
    dataMat = []
    lableMat = []
    with open("./dataSource/testSet.txt") as dataFile:

        for line in dataFile.readlines():
            lineArr = line.strip().split()
            # 这里添加的1.0是为了将公式中的偏执项作为一个w0来处理。
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            lableMat.append(int(lineArr[2]))

        return dataMat, lableMat


def sigmoid(intX):
    return 1.0 / (1 + exp(-intX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    lableMat = mat(classLabels).transpose()
    # shape 函数获取矩阵的行数m和列数n
    m, n = shape(dataMatrix)
    # alpha 代表步长
    alpha = 0.001
    maxCycles = 500
    # ones 第一个参数是 shape 生成矩阵的行和列数。这里就是初始化一个权重是1的列矩阵出来
    weights = ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (lableMat - h)
        # 更新权重
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


if __name__ == '__main__':
    dataArr, lableMat = loadDataSet()
    weights = gradAscent(dataArr, lableMat)
    print(weights)
