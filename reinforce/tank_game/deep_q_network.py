#!/usr/bin/env python
# -*-coding:UTF-8 -*-

import numpy as np 

from tanksbattle import tanks

ACTIONS = 6


if __name__ == '__main__':

    game_state = tanks.GameState()
    #do_nothing = np.zeros(ACTIONS)
    #do_nothing[0] = 1

    do_action = np.zeros(ACTIONS)
    do_action[1] = 1#fire

    for i in range(1500):       
        x_t, r_0, terminal = game_state.frame_step(do_action)
        print('x_t shape:{} reward:{} terminal:{}'.format(np.shape(x_t), r_0, terminal))





