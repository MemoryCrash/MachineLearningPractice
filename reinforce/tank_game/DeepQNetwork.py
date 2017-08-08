#!/usr/bin/env python
# -*-coding:UTF-8 -*-

import cv2
import random
import numpy as np 
import tensorflow as tf 
from collections import deque

from tanksbattle import Tanks

def image_chg(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r.transpose(), g.transpose(), b.transpose()])

class DeepQNetwork:

    def _network(self, c_names):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) 
        
        conv2d = lambda x, W, stride: tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
        max_pool_2x2 = lambda x: tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        # 卷积层1
        with tf.variable_scope('conv1'):
            wc1 = tf.get_variable('w_c1', [8, 8, 4, 32], initializer=w_initializer, collections=c_names)
            bc1 = tf.get_variable('b_c1', [32], initializer=b_initializer, collections=c_names)
            h_conv1 = tf.nn.relu(conv2d(self.s, wc1, 4) + bc1)
            h_pool1 = max_pool_2x2(h_conv1)

        # 卷积层2
        with tf.variable_scope('conv2'):
            wc2 = tf.get_variable('w_c2', [4, 4, 32, 64], initializer=w_initializer, collections=c_names)
            bc2 = tf.get_variable('b_c2', [64], initializer=b_initializer, collections=c_names)
            h_conv2 = tf.nn.relu(conv2d(h_pool1, wc2, 2) + bc2)

        # 卷积层3
        with tf.variable_scope('conv3'):
            wc3 = tf.get_variable('w_c3', [3, 3, 64, 64], initializer=w_initializer, collections=c_names)
            bc3 = tf.get_variable('b_c3', [64], initializer=b_initializer, collections=c_names)
            h_conv3 = tf.nn.relu(conv2d(h_conv2, wc3, 1) + bc3)           
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600]) 

        # 第一层. collections 是在更新 target_net 参数时会用到
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [1600, 512], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [512], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(h_conv3_flat, w1) + b1)

        # 第二层. collections 是在更新 target_net 参数时会用到
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [512, self.n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [self.n_actions], initializer=b_initializer, collections=c_names)
            value = tf.matmul(l1, w2) + b2

        return value


    def _build_net(self):
        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        self.s = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 4], name='s')  # 用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # 用来接收 q_target 的值, 这个之后会通过计算得到
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = self._network(c_names)

        with tf.variable_scope('loss'): # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):    # 梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 4], name='s_')    # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = self._network(c_names)


    def __init__(
        self,
        n_actions,
        image_size,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=1000,
        memory_size=200,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False,
):
        self.n_actions = n_actions
        self.image_size = image_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy     # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size    # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment # epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max # 是否开启探索模式, 并逐步减少探索次数

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # 记忆 [s, a, r, s_, t]
        self.memory = deque()
        # 创建 [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs_tank/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看
        self.net_restore()

    def store_transition(self, s, a, r, s_, t):

        # 记录一条 [s, a, r, s_] 记录
        self.memory.append((s, a, r, s_, t))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()



    def choose_action(self, observation, step, OBSERVE):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        action = np.zeros(self.n_actions)

        if np.random.uniform() < self.epsilon and step > OBSERVE:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

            action[np.argmax(actions_value)]=1
        else:
            action[np.random.randint(0, self.n_actions)] = 1   # 随机选择
            #print('random choice:{}'.format(step))

        return action


    def _replace_target_params(self):
        # 使用 Tensorflow 中的 assign 功能替换 target_net 所有参数
        t_params = tf.get_collection('target_net_params')   # 提取 target_net 的参数
        e_params = tf.get_collection('eval_net_params') # 提取  eval_net 的参数
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])  # 更新 target_net 参数

    def learn(self):
        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget_params_replaced\n')

        # 从 memory 中随机抽取 batch_size 这么多记忆
        batch_memory = random.sample(self.memory, self.batch_size)
        batch_s = [d[0] for d in batch_memory]
        batch_a = [d[1] for d in batch_memory]
        batch_r = [d[2] for d in batch_memory]
        batch_s_ = [d[3] for d in batch_memory]

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_s_,
                self.s: batch_s
            })

        q_target = q_eval.copy()
        for i in range(0, len(batch_memory)):
            if batch_memory[i][4]:
                q_target[i][np.argmax(batch_a[i])] = batch_r[i]
            else:
                q_target[i][np.argmax(batch_a[i])] \
                = batch_r[i] + self.gamma * np.max(q_next[i])

        #eval_act_index = np.argmax(batch_a, axis=1)
        #reward = batch_r
        #batch_index = np.arange(self.batch_size, dtype=np.int32)
        #q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        # 训练 eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_s,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost) # 记录 cost 误差

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def net_saver(self, step):
        if step % 50000 == 0:
            self.saver.save(self.sess, 'saved_networks_tank/' + 'tank-dqn', global_step = step)

    def net_restore(self):
        checkpoint = tf.train.get_checkpoint_state("saved_networks_tank")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")





