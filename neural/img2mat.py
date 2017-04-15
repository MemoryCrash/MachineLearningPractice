#!/usr/bin/env python
# -*-coding:UTF-8 -*-
from PIL import Image
import numpy as np
import struct


def rgb2gray(imageFile):
    rgbImage = Image.open(imageFile)

    grayImg = rgbImage.convert('L')
    grayMat = np.array(grayImg.getdata())

    data = (np.reshape(grayMat, (784, 1))) / 255

    return data


def read_image(filename):
    with open(filename, 'rb') as f:
        buf = f.read()
    index = 0

    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        image = Image.new('L', (columns, rows))

        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        print('save {} image'.format(i))
        image.save('test/' + str(i) + '.png')



if __name__ == '__main__':
    grayData = rgb2gray('./dataNum/2A.png')
    print(grayData)
    #read_image('t10k-images.idx3-ubyte')
