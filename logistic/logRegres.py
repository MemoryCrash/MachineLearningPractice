#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt


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


def plotBestFit(weights):
    dataMat, lableMat = loadDataSet()
    # 当需要使用多维数组做大计算的时候最好将 python 本身的 list 类型使用 numpy 的 array
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(lableMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    # 创建一个图表对象
    fig = plt.figure()
    # 将画布分为1行1列并使用从左到右从上到下数的第1个
    ax = fig.add_subplot(111)
    # scatter 散点图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 划分类别的曲线
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr, lableMat = loadDataSet()
    weights = gradAscent(dataArr, lableMat)
    print(weights)
    # 这里 getA() 的作用是将矩阵返回为一个多维数组
    plotBestFit(weights.getA())
