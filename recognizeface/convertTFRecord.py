#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 

import cv2
import time
import numpy as np 
import tensorflow as tf
from datetime import timedelta
from getFace import print_image_progress

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 

def load_labels(filepath):
    """depend file name get label info"""
    labels = {}
    for (path, dirnames, filenames) in os.walk(filepath):
        if len(dirnames):
            for dirInfo in dirnames:
                labelInfo=dirInfo.split('_')
                labels[dirInfo] = int(labelInfo[1])

    return labels

def extract_images(filename):
    """f is file handle"""

    #image = [rows,cols,channels]
    return cv2.imread(filename)





def convert2tf(input_dir, output_dir, output_file):
    start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = "{}/{}.tfrecords".format(output_dir, output_file)
    writer = tf.python_io.TFRecordWriter(output_path)
    labels = load_labels(input_dir)

    for (path, dirnames, filenames) in os.walk(input_dir):
        curdir = path.split('/')[-1] 
        if curdir in labels.keys():
            label = int(labels[curdir])
            SUM_NUMBER = len(filenames)
            index = 1
            print('\n')

            for filename in filenames:
                if filename.endswith('.jpg'):
                    filepath = os.path.join(path, filename)
                    image = extract_images(filepath)
                    image_raw = image.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={  
                        'image_raw': _bytes_feature(image_raw),  
                        'height': _int64_feature(image.shape[0]),  
                        'width': _int64_feature(image.shape[1]),  
                        'depth': _int64_feature(image.shape[2]),  
                        'label': _int64_feature(label)  
                    }))  
                    writer.write(example.SerializeToString())

                    print_image_progress(index, 1, SUM_NUMBER, start_time)
                    index += 1 

    writer.close()

    end_time = time.time()
    time_dif = end_time - start_time
    print("\n- Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

if __name__ == '__main__':

    input_dir = '/home/data/recognizeface/train_data'
    output_dir = '/home/data/recognizeface/tf_data'
    output_file = 'face_train'

    convert2tf(input_dir, output_dir, output_file)