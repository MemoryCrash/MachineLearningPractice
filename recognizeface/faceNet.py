#!/usr/bin/env python
# -*-coding:UTF-8 -*-
import os 
import sys 

import cv2
import time
import numpy as np 
import tensorflow as tf
import prettytensor as pt
from datetime import timedelta

from PIL import Image
from getFace import print_image_progress
from readTFRecord import read_and_decode, read_all_records
from getFace import getSingleImage
import FLAG

def main_network(images, training):
    x_pretty = pt.wrap(images)

    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=FLAG.NUM_CLASSES, labels=y_true)

    return y_pred, loss


def create_network(training):
    with tf.variable_scope('network', reuse=not training):
        images = x
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss

def loadCheckPoint(session):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'face_cnn')
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    try:
        print("Trying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

        saver.restore(session, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(init)


def dense_to_one_hot(labels_dense, num_classes):
    """convert class labels from scalars to one-hot vertors"""

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def optimize(session):
    # set tfrecords file 
    sparse_images, sparse_labels = read_and_decode(FLAG.FILE_TRAIN_NAME, 60, 10000, 9000)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    
    for i in range(FLAG.NUM_ITERATIONS):
        images, labels = session.run([sparse_images, sparse_labels])
        labels_one_hot = dense_to_one_hot(labels, FLAG.NUM_CLASSES)
        feed_dict_train = {x: images, y_true: labels_one_hot}

        i_global, _ = session.run([global_step, optimizer],feed_dict=feed_dict_train)

        if (i_global % 100 == 0) or (i == FLAG.NUM_ITERATIONS - 1):
            batch_acc = session.run(accuracy,feed_dict=feed_dict_train)
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        if (i_global % 1000 == 0) or (i == FLAG.NUM_ITERATIONS - 1):
            saver.save(session, save_path=save_path, global_step=global_step)
            print("Saved checkpoint.")


    coord.request_stop()
    coord.join(threads)


def test_accuracy(session):

    sparse_images, sparse_labels = read_all_records(FLAG.FILE_TEST_NAME, 2300)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)


    images, labels = session.run([sparse_images, sparse_labels])    

    labels_one_hot = dense_to_one_hot(labels, FLAG.NUM_CLASSES)
    feed_dict = {x: images, y_true: labels_one_hot}

    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    correct = (labels == cls_pred)

    acc = correct.mean()
    num_correct = correct.sum()
    
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    coord.request_stop()
    coord.join(threads)

def faceRec(session):
    print("Begin face Recognize...")
    labelName = ['OTHERS','DAI LEI', 'HUANGQIN']
    face_raw = getSingleImage()
    #给np array 添加一个维度由[height, width, num_channels]
    #变为[img_num, height, width, num_channels]
    face = np.float32(face_raw) * (1. / 255) - 0.5
    img = face[np.newaxis,:]
    feed_dict = {x: img}
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    print('\n Is [{}]\n'.format(labelName[cls_pred[0]]))

    face_raw = Image.fromarray(face_raw)
    face_raw.show()



# set input variable
x = tf.placeholder(tf.float32,shape=[None, FLAG.IMAGE_ROW, FLAG.IMAGE_COL, FLAG.IMAGE_CHANNELS], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, FLAG.NUM_CLASSES], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# create train network
_, loss = create_network(training=True)
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# create test network
y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create train model saver
saver = tf.train.Saver()
save_dir = 'checkpoints/'
save_path = os.path.join(save_dir, 'face_cnn')

with tf.Session() as sess:

    loadCheckPoint(sess)

    #1# optimize(sess)

    #2# test_accuracy(sess)

    faceRec(sess)


