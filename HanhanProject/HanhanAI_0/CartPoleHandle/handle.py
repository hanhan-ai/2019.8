"""
作者：赵士陆
创建时间：2019.8.24
最后一次修改时间：2019.8.30

"""
#打飞机游戏处理

import gym
import numpy as np
import cv2
# from DeepQNetwork.Network_2 import *

env = gym.make('CartPole-v0')

action_number = env.action_space
stay_action = 1     # 暂定为1
def CartPolePictureHandle(observation):     # Process the picture of the game as the input of the network
    cp_observation_origin = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # cp_observation = cv2.resize(cp_observation_origin, (inputImageSize[1], inputImageSize[0]))
    print("def CartPolePictureHandle(observation)","cp_observation_origin:  ",cp_observation_origin)
    return cp_observation_origin


def CartPoleActionHandle(act):              # Process the action of the game as the output of the network
    for k in range(0, len(act)):
        if act[k] == 1:
            act_return = k
        else:
            pass
    print("def CartPoleActionHandle(act)", "act_return:  ", act_return)
    return act_return

def Game():                                 # Game cycle

        ob = env.reset()

        while True:
            env.render()
            print(ob)

            # do action
            action = env.action_space.sample()

            # input action to the environment and handle these returned parameters
            ob, reward, done, info = env.step(action)

            # picture handles
            input_image = CartPolePictureHandle(ob)

            if done:
                print("Episode finished after {} timesteps".format(1))
                break
