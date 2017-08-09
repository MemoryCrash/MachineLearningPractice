#!/usr/bin/python
# coding=utf-8
import sys

import cv2
import time
import numpy as np
from datetime import timedelta

from tanksbattle import Tanks
from DeepQNetwork import DeepQNetwork, image_chg

ACTIONS = 6
IMAGE_SIZE = 80
OBSERVE = 5000
EXPLORE = 10000.


def print_progress(episode, episode_max, start_time):
    """
    打印花费的时间以及显示玩了多少回合
    """
    current_time = time.time()
    time_dif = current_time - start_time
    use_time = str(timedelta(seconds=int(round(time_dif))))

    msg = "\r- episode:{}/{} - Use time:{}"
    msg = msg.format(episode, episode_max, use_time)

    sys.stdout.write(msg)
    sys.stdout.flush()


def env_reset(env):
    """
    给出游戏开始初始的第一个环境图片
    """
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = env.frame_step(do_nothing)
    x_t = image_chg(x_t)

    x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
    x_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # 输入的数据进行归一化，均值设置为0
    return x_t / 255.0


def next_observation(raw_observation_, observation):
    """
    对动作作用后的原始的环境帧图片进行处理，将上个环境图片添加三张进去
    """
    raw_observation_ = cv2.cvtColor(raw_observation_, cv2.COLOR_BGR2GRAY)
    # 输入的数据进行归一化，均值设置为0
    raw_observation_ = raw_observation_ / 255.0

    raw_observation_ = np.reshape(raw_observation_, (80, 80, 1))
    return np.append(raw_observation_, observation[:, :, 0:3], axis=2)


def run_tanks():
    """
    运行坦克大战游戏
    """
    step = 0
    state_num = 0
    play_time = 2000

    for episode in range(play_time):
        print_progress(episode, play_time, start_time)
        observation = env_reset(env)
        episode_reward = 0

        while True:
            # 选择动作
            action = RL.choose_action(observation, state_num, OBSERVE, EXPLORE)

            # 施加选择的动作到环境上，环境返回动作作用后环境信息，当前动作奖励，环境状态(结束true/继续false)
            raw_observation_, reward, terminal = env.frame_step(action)
            raw_observation_ = image_chg(raw_observation_)
            observation_ = next_observation(raw_observation_, observation)

            # 平衡下存储的信息，让reward大的多放点在memory中。
            if np.random.uniform() < 0.8:
                if abs(reward) > 5:
                    # 将以上的环境，动作，动作反馈，动作作用后的环境，环境状态存入队列，后续训练使用
                    RL.store_transition(observation, action, reward, observation_, terminal)
                    state_num += 1
            else:
                # 将以上的环境，动作，动作反馈，动作作用后的环境，环境状态存入队列，后续训练使用
                RL.store_transition(observation, action, reward, observation_, terminal)
                state_num += 1
            #print(state_num)

            if state_num == OBSERVE:
                print('\nMemory is enough begin to train')

            # 先运行一段时间收集足够的训练信息再开始学习
            if(state_num > OBSERVE) and (step % 5 == 0):
                RL.learn()

            # 保存训练的神经网络
            RL.net_saver(step)

            # 将环境进行转换，动作作用后的环境变成了当前环境
            observation = observation_
            episode_reward += reward

            if terminal:
                break

            step += 1

        print('\n episode:{}  reward:{}'.format(episode, episode_reward))


if __name__ == '__main__':
    start_time = time.time()
    env = Tanks.GameState()

    RL = DeepQNetwork(ACTIONS, IMAGE_SIZE,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=5000,
        memory_size=50000,
        output_graph=True
        )

    run_tanks()
    print('\n')
