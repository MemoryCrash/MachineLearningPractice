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


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    lableMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros(m, 1))
    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, lableMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(lableMat[i])
            if((lableMat[i] * Ei < -toler) and (alphas[i] < C)) or ((lableMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, lableMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(lableMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (lableMat[i] != lableMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print("L==H")
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                dataMatrix[i, :] * dataMatrix[i].T - \
                dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:
                    print("eta>=0")
                    continue

                alphas[i] -= lableMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue

                alphas[i] += lableMat[j] * lableMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - lableMat[i] * (alphas[i] - alphaIold) * \
                dataMatrix[i, :] * dataMatrix[i, :].T - \
                lableMat[j] * (alphas[j] - alphaJold) * \
                dataMatrix[i, :] * dataMatrix[j, :].T

                b2 = b - Ej - lableMat[i] * (alphas[i] - alphaIold) * \
                dataMatrix[i, :] * dataMatrix[j, :].T - \
                lableMat[j] * (alphas[j] - alphaJold) * \
                dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j] and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1 + b2)/2.0

                alphaPairsChanged += 1
                print("iter:%d i :%d, pairs changed %d"% (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0

        print("iteration number:%d"%iter)

    return b, alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet("./dataSource/testSet.txt")
    print(labelArr)

