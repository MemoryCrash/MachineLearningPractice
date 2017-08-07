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
        return image_chg(x_t) 

def run_tanks():
    step = 0
    play_time = 10

    for episode in range(play_time):
        print_progress(episode, play_time, start_time)
        observation = env_reset(env)

        while True:
            action = RL.choose_action(observation)
            observation_, reward, terminal = env.frame_step(action)

            RL.store_transition(observation, action, reward, image_chg(observation_), terminal)

            if(step > 200) and (step % 5 == 0):
                RL.learn()

            RL.net_saver(step)

            observation = observation_

            if terminal:
                break

            step += 1


if __name__ == '__main__':
    start_time = time.time()
    env = Tanks.GameState()

    RL = DeepQNetwork(ACTIONS, IMAGE_SIZE,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=200,
        output_graph=True
        )

    run_tanks()
    #RL.plot_cost()
    print('\n')


