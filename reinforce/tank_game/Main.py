#!/usr/bin/python
# coding=utf-8
import cv2
import numpy as np 
from tanksbattle import Tanks
from DeepQNetwork import DeepQNetwork, image_chg

ACTIONS = 6
IMAGE_SIZE = 80

def env_reset(env):
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = env.frame_step(do_nothing)
        return image_chg(x_t) 

def run_tanks():
    step = 0
    for episode in range(10):
        observation = env_reset(env)

        while True:
            action = RL.choose_action(observation)
            observation_, reward, terminal = env.frame_step(action)

            RL.store_transition(observation, action, reward, image_chg(observation_))

            if(step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if terminal:
                break

            step += 1


if __name__ == '__main__':
    env = Tanks.GameState()

    RL = DeepQNetwork(ACTIONS, IMAGE_SIZE,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=200
        )

    run_tanks()



