#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 
import random

import dlib
import cv2
import shutil
import numpy as np 
import time
import tensorflow as tf
from datetime import timedelta

from getFace import print_image_progress
import FLAG

if not os.path.exists(FLAG.SAVE_FACE_PATH):
	os.makedirs(FLAG.SAVE_FACE_PATH)

detector = dlib.get_frontal_face_detector()

def moveImage():

    for (path, dirnames, filenames) in os.walk(FLAG.DLD_IMAGE_PATH):
        for filename in filenames:
            if filename.endswith('.jpg'):
                img_path = path+'/'+filename
                shutil.copy(img_path, FLAG.RAW_IMAGE_PATH)



def getFace():

    start_time = time.time()
    index = 0

    for (path, dirnames, filenames) in os.walk(FLAG.INPUT_FACE_PATH):
        IMAGE_NUMBER = len(filenames)
        for filename in filenames:	
            if filename.endswith('.jpg'):
                img_path = path+'/'+filename

                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_img, 1)

                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0

                    face = img[x1:y1, x2:y2]
                    face = cv2.resize(face, (FLAG.IMAGE_SIZE, FLAG.IMAGE_SIZE))
                    cv2.imwrite(FLAG.SAVE_FACE_PATH+'/'+str(index)+'.jpg', face)
                    index += 1
                    print_image_progress(index, 1, IMAGE_NUMBER, start_time)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print(key)
                    break

    end_time = time.time()
    time_dif = end_time - start_time
    print("\n- Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

if __name__ == '__main__':
	getFace()