#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from PIL import Image
import numpy as np


def rgb2gray(imageFile):
    rgbImage = Image.open(imageFile)

    grayImg = rgbImage.convert('L')
    grayMat = np.matrix(grayImg.getdata())

    data = (255 - np.reshape(grayMat, (784, 1))) / 255

    return data


if __name__ == '__main__':
    grayData = rgb2gray('2.png')
    print(grayData)
