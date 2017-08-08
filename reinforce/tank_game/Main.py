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
OBSERVE = 10000

def print_progress(episode, episode_max, start_time):
    current_time = time.time()
    time_dif = current_time - start_time
    use_time = str(timedelta(seconds=int(round(time_dif))))
    msg = "\r- episode:{}/{} - Use time:{}".format(episode, episode_max, use_time)

    sys.stdout.write(msg)
    sys.stdout.flush()

def env_reset(env):
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = env.frame_step(do_nothing)
        x_t = image_chg(x_t)

        x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        return np.stack((x_t, x_t, x_t, x_t), axis = 2)

def next_observation(raw_observation_, observation):
    raw_observation_ = cv2.cvtColor(raw_observation_, cv2.COLOR_BGR2GRAY)
    ret, raw_observation_ = cv2.threshold(raw_observation_,1,255,cv2.THRESH_BINARY)
    raw_observation_ = np.reshape(raw_observation_, (80, 80, 1))

    return np.append(raw_observation_, observation[:,:,0:3], axis = 2)


def run_tanks():
    step = 0
    play_time = 1000

    for episode in range(play_time):
        print_progress(episode, play_time, start_time)
        observation = env_reset(env)
        episode_reward = 0

        while True:

            action = RL.choose_action(observation, step, OBSERVE)

            raw_observation_, reward, terminal = env.frame_step(action)
            episode_reward += reward

            raw_observation_ = image_chg(raw_observation_)
            observation_ = next_observation(raw_observation_, observation)

            RL.store_transition(observation, action, reward, observation_, terminal)

            if step == OBSERVE:
                print('\nMemory is enough begin to train')

            if(step > OBSERVE) and (step % 5 == 0):
                RL.learn()

            RL.net_saver(step)

            observation = observation_

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
        replace_target_iter=800,
        memory_size=10000,
        output_graph=True
        )

    run_tanks()
    #RL.plot_cost()
    print('\n')


