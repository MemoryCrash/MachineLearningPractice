#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(x) for x in curLine]
        dataMat.append(fltLine)

    fr.close()

    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 生成一个k行1列的0到1之间的随机数
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)

    return centroids



def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if(distJI < minDist):
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
            # print(centroids)

        for cent in range(k):
            # 将某个类的数据的行找出来，并对应行找到数据。
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # axis=0表示根据矩阵的列方向来求均值
            centroids[cent, :] = mean(ptsInClust, axis=0)

    return centroids, clusterAssment


if __name__ == '__main__':

    datMat = mat(loadDataSet('./dataSource/testSet.txt'))
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids)

