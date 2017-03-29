#!/usr/bin/env python
# -*-coding:UTF-8 -*-

from numpy import *
import re

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', \
                    'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
                    'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],]
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:{} is not in my Vocabulaty!".format(word))

    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = {'love', 'my', 'dalmation'}
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    result = classifNB(thisDoc, p0V, p1V, pAb)
    print("{} classified as:{}".format(testEntry, result))

    testEntry = {'stupid', 'garbage'}
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    result = classifNB(thisDoc, p0V, p1V, pAb)
    print("{} classified as:{}".format(testEntry, result))


def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)

    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    # 这里这样写是因为有26个文件，但是其实可以遍历这个目录下的所有文件
    for i in range(1, 26):
        file = open('./dataSource/email/spam/%d.txt' % i)
        wordList = textParse(file.read())
        docList.append(wordList)
        classList.append(1)
        file.close()

        file = open('./dataSource/email/ham/%d.txt' % i, errors = 'ignore')
        wordList = textParse(file.read())
        docList.append(wordList)
        classList.append(0)
        file.close()

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []

    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        result = classifNB(array(wordVector), p0V, p1V, pSpam)
        if result != classList[docIndex]:
            errorCount += 1

    print("the error rate is : {}".format((errorCount) / len(testSet)))


if __name__ == '__main__':
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)

    # trainMat = []
    # for postinDoc in listOPosts:
        # trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(p0V, p1V, pAb)
    # testingNB()
    spamTest()
