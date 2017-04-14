#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from PIL import Image
import numpy as np


def rgb2gray(rgbImage):

    grayImg = rgbImage.convert('L')
    grayMat = np.matrix(grayImg.getdata())

    data = (255 - np.reshape(grayMat, (784, 1))) / 255

    return data


if __name__ == '__main__':
    rgbImage = Image.open('2.png')
    grayData = rgb2gray(rgbImage)
    print(grayData)
