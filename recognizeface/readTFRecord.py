#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 

import cv2
import time
import numpy as np 
import tensorflow as tf

import FLAG

def read_and_decode(file_name, batch_size, capacity, min_after_dequeue):

    filename_queue = tf.train.string_input_producer([file_name])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, 
                                       features={
                                       'image_raw': tf.FixedLenFeature([], tf.string),
                                       'label': tf.FixedLenFeature([], tf.int64)
                                       })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [FLAG.IMAGE_ROW, FLAG.IMAGE_COL, FLAG.IMAGE_CHANNELS])
    image = tf.cast(image, tf.float32)*(1. / 255) - 0.5
    #[example,label],capacity:shuffle range,batch_size:sample number,
    #min_after_dequeue:min_after_dequeue should less than capacity when queue greater than it then shuffle
    images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,  min_after_dequeue=min_after_dequeue)

    return images, sparse_labels

def read_all_records(file_name, img_size):

    filename_queue = tf.train.string_input_producer([file_name])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, 
                                       features={
                                       'image_raw': tf.FixedLenFeature([], tf.string),
                                       'label': tf.FixedLenFeature([], tf.int64)
                                       })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [FLAG.IMAGE_ROW, FLAG.IMAGE_COL, FLAG.IMAGE_CHANNELS])
    image = tf.cast(image, tf.float32)*(1. / 255) - 0.5
    #[example,label],capacity:shuffle range,batch_size:sample number,
    #min_after_dequeue:min_after_dequeue should less than capacity when queue greater than it then shuffle
    images, sparse_labels = tf.train.batch([image, label], batch_size=img_size, capacity=img_size)

    return images, sparse_labels



if __name__ == '__main__':
    pass