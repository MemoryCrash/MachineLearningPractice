#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 
import random

import cv2
import dlib
import time
import tensorflow as tf
from datetime import timedelta

import FLAG 

#设置保存路径
if not os.path.exists(FLAG.SAVE_FACEIMG_PATH):
    os.makedirs(FLAG.SAVE_FACEIMG_PATH)

#打印图片处理进度
def print_image_progress(count, block_size, total_size, start_time):

    pct_complete = float(count * block_size) / total_size

    current_time = time.time()
    time_dif = current_time - start_time
    use_time = str(timedelta(seconds=int(round(time_dif))))
    msg = "\r- Image progress: {0:.1%} -Use time:{1}".format(pct_complete, use_time)

    sys.stdout.write(msg)
    sys.stdout.flush()

def getImage():

    index = 0
    start_time = time.time()
    #使用opencv打开电脑摄像头，设置参数为输入流(摄像头、视频文件)
    camera = cv2.VideoCapture(0)
    #使用dlib的frontal_face_detector作为特征提取器(获取人脸头像)
    detector = dlib.get_frontal_face_detector()

    while(index < IMAGE_NUMBER):
        _, img = camera.read()
        # 将图片转换为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用灰度图片进行人脸检测
        dets = detector(gray_img, 1)
        # 这里返回的dets表示检测到的人脸的数量集合。d中包含了以下位置信息。
        #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离 
        #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            face = cv2.resize(face, (FLAG.IMAGE_ROW, FLAG.IMAGE_COL))

            cv2.imwrite(SAVE_FACE_PATH+'/'+str(index)+'.jpg', face)

            print_image_progress(index+1, 1, FLAG.IMAGE_NUMBER, start_time)
            index += 1
        #waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。
        #返回值为当前键盘按键值
        key = cv2.waitKey(30) & 0xFF
        #如果输入ESC就停止。这里27表示的是ESC键
        if key == ord('q'):
            print(key)
            break

    end_time = time.time()
    time_dif = end_time - start_time
    print("\n- Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def getSingleImage():

    while True:
        #使用opencv打开电脑摄像头，设置参数为输入流(摄像头、视频文件)
        camera = cv2.VideoCapture(0)
        #使用dlib的frontal_face_detector作为特征提取器(获取人脸头像)
        detector = dlib.get_frontal_face_detector()
        _, img = camera.read()
        # 将图片转换为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用灰度图片进行人脸检测
        dets = detector(gray_img, 1)
        # 这里返回的dets表示检测到的人脸的数量集合。d中包含了以下位置信息。
        #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离 
        #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            face = cv2.resize(face, (FLAG.IMAGE_ROW, FLAG.IMAGE_COL))

            return face

if __name__ == '__main__':
    getImage()





