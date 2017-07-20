#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 
import random

import shutil

def splitTrainTest(input_dir, test_dir, image_num):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    testImageIndex = 0
    testImageSum = round(image_num / 10)

    while testImageIndex <= testImageSum:

        imageIndex = random.randint(0, image_num-1)
        imageFile = str(imageIndex)+'.jpg'

        src = os.path.join(input_dir, imageFile)
        dst = os.path.join(test_dir, imageFile)
        if os.path.exists(src):
            shutil.move(src, dst)
            testImageIndex += 1



if __name__ == '__main__':

    input_dir = '/home/data/recognizeface/data/other_0'
    test_dir = '/home/data/recognizeface/test_data/other_0'
    image_num = 13982

    splitTrainTest(input_dir, test_dir, image_num)

