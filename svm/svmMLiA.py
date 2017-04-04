#!/usr/bin/env python
# -*-coding:UTF-8 -*-

from numpy import *


def loadDataSet(fileName):
    dataMat = []
    lableMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        lableMat.append(float(lineArr[2]))

    fr.close()

    return dataMat, lableMat


def selectJrand(i, m):
    j = i

    while (j == i):
        j = int(random.uniform(0, m))

    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet("./dataSource/testSet.txt")
    print(labelArr)

