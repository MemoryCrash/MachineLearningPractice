#!/usr/bin/env python
# -*-coding:UTF-8 -*-

from numpy import *


# 加载训练数据
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


# 随机选择第二个待优化点
def selectJrand(i, m):
    j = i

    while (j == i):
        j = int(random.uniform(0, m))

    return j


# 对优化后的数据进行裁剪
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# smo简易实现
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    lableMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, lableMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(lableMat[i])
            if((lableMat[i] * Ei < -toler) and (alphas[i] < C)) or ((lableMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, lableMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
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

                alphas[j] -= lableMat[j] * (Ei - Ej) / eta
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


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.lableMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.lableMat).T * oS.K[:, k])+ oS.b
    Ek = fXk - float(oS.lableMat[k])
    return Ek


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    # 将输入的数字的非零的位置返回，反化一个元组，对于二维数据，元组中的数据就包含了行位置的列表和列位置的列表
    # matrix.A 返回一个 ndarray 的自己
    # [:,0]返回第一列的数据
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]

    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)

    return j, Ej


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (((oS.lableMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
     ((oS.lableMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0))):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if(oS.lableMat[i] != oS.lableMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if(L == H):
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]

        if eta >= 0:
            print("eta >= 0")
            return 0

        oS.alphas[j] -= oS.lableMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)

        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0

        oS.alphas[i] += oS.lableMat[i] * oS.lableMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.lableMat[i] * (oS.alphas[i] - alphaIold) * \
        oS.K[i, i] - oS.lableMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]

        b2 = oS.b - Ej - oS.lableMat[i] * (oS.alphas[i] - alphaIold) * \
        oS.K[i, j] - oS.lableMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        if(0 < oS.alphas[i] and oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0 < oS.alphas[j] and oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2

        return 1

    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    while(iter < maxIter and alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if(entireSet):
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter:%d i:%d, paris changed %d" %(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter:%d i:%d, paris changed %d"%(iter, i, alphaPairsChanged))
            iter += 1
        if(entireSet):
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number:%d"% iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    lableMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))

    for i in range(m):
        w += multiply(alphas[i] * lableMat[i], X[i, :].T)

    return w


def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif(kTup[0] == 'rbf'):
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem That Kernel is not recognized')

    return K



if __name__ == '__main__':
    dataArr, labelArr = loadDataSet("./dataSource/testSet.txt")

    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas>0])
    ws = calcWs(alphas, dataArr, labelArr)

    datMat = mat(dataArr)
    print(datMat[0] * mat(ws) + b)
    print(labelArr[0])
    print(datMat[1] * mat(ws) + b)
    print(labelArr[1])
    print(datMat[2] * mat(ws) + b)
    print(labelArr[2])
