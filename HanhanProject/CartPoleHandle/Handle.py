import gym
import numpy as np
import cv2
# from DeepQNetwork.Network_2 import *

env = gym.make('CartPole-v0')

action_number = env.action_space
stay_action = 1     # 其实不是
def CartPolePictureHandle(observation):
    cp_observation_origin = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # cp_observation = cv2.resize(cp_observation_origin, (inputImageSize[1], inputImageSize[0]))
    return cp_observation_origin


def CartPoleActionHandle(act):
    for k in range(0, len(act)):
        if act[k] == 1:
            act_return = k
        else:
            pass
    return act_return

def Game():

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
