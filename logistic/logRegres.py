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

# 随机梯度上升或者下降算法，当面对数据比较大的时候像是我们刚才的梯度上升算法
# 就显的非常的慢了假设我们有1w条数据那么我们要更新一次数据就需要把这1w条数据
# 都做一次计算。所以为了加快数据更新的速度这里实现一种叫做随机梯度上升的算法


def stocGradAscent0(dataMatrix, classLabels):
    '''
    这个函数目前的运行效果实际来看并不怎么好，当然有个原因是目前我们的数据量少
    '''
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones(n)

    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]

    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex = list(range(m))

        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]

    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('./dataSource/horseColicTraining.txt')
    frTest = open('./dataSource/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))

        hResult = int(classifyVector(array(lineArr), trainWeights))
        if hResult != int(currLine[21]):
            errorCount += 1

    frTrain.close()
    frTest.close()

    errorRate = float(errorCount) / numTestVec
    print("the error rate of this test is:%f" % errorRate)

    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0

    for k in range(numTests):
        errorSum += colicTest()

    msg = "after %d iterations the average error rate is %f"
    print(msg % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    # dataArr, lableMat = loadDataSet()
    # weights = gradAscent(dataArr, lableMat)
    # weights = stocGradAscent1(array(dataArr), lableMat)
    # print(weights)
    # 这里 getA() 的作用是将矩阵返回为一个多维数组
    # plotBestFit(weights.getA())
    # plotBestFit(weights)
    multiTest()
