#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 
import random

import cv2
import numpy as np 
import time
import tensorflow as tf
from datetime import timedelta

from getFace import print_image_progress
import FLAG



#帮助函数，改变图片参数来增加样本多样性。
def pre_process_image(image):
    #随机左右翻转图像
    #image = tf.image.flip_left_right(image)
    #image = tf.image.random_flip_left_right(image)
    #随机色调调整
    #image = tf.image.adjust_hue(image, delta=0.05)
    image = tf.image.random_hue(image, max_delta=0.05)
    #随机对比度调整
    #image = tf.image.adjust_contrast(image, contrast_factor=1.5)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    #随机亮度调整
    #image = tf.image.adjust_brightness(image, delta=0.5)
    image = tf.image.random_brightness(image, max_delta=0.2)
    #随机饱和度调整
    #image = tf.image.adjust_saturation(image, saturation_factor=2)
    #image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    #截断值在0到255之间
    image = tf.minimum(image, 255.0)
    image = tf.maximum(image, 0.0)

    return image

def increase_image():
    
    sess = tf.Session()
    start_time = time.time()
    index = 0
    
    for (path, dirnames, filenames) in os.walk(FLAG.INPUT_FACE_PATH_INC):
        
        IMAGE_NUMBER = len(filenames)

        for filename in filenames:
            if filename.endswith('.jpg'):
                img_path = path+'/'+filename
                img = cv2.imread(img_path)
                # trensorflow没有uint8类型的变量所以需要进行类型转换
                img = tf.convert_to_tensor(img, dtype=tf.float32)
                increase_img = pre_process_image(img)
                cv2.imwrite(FLAG.SAVE_FACE_PATH_INC+'/'+str(index)+'inc2.jpg', increase_img.eval(session=sess))
                index += 1

                print_image_progress(index, 1, IMAGE_NUMBER, start_time)


            key = cv2.waitKey(30) & 0xFF
            #如果输入ESC就停止。这里27表示的是ESC键
            if key == ord('q'):
                print(key)
                break

    end_time = time.time()
    time_dif = end_time - start_time
    print("\n- Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


if __name__ == '__main__':
    increase_image()



