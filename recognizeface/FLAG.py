#!/usr/bin/env python
# -*-coding:UTF-8 -*-

IMAGE_ROW = 64
IMAGE_COL = 64
IMAGE_CHANNELS = 3
NUM_CLASSES = 3

IMAGE_NUMBER = 10000

NUM_ITERATIONS = 1000
NUM_ITERATIONS_TEST = 1

SAVE_FACEIMG_PATH = './dst_faces'

INPUT_FACE_PATH = './data/rawFace/face'
SAVE_FACE_PATH = './data/other_0'

DLD_IMAGE_PATH = './data/rawFace/lfw'
RAW_IMAGE_PATH = './data/rawFace/face'

INPUT_FACE_PATH_INC = './dst_faces'
SAVE_FACE_PATH_INC = './dst_faces_inc'

FILE_TRAIN_NAME = '/home/data/recognizeface/tf_data/face_train.tfrecords' 
FILE_TEST_NAME = '/home/data/recognizeface/tf_data/face_test.tfrecords' 

