"""
Deep Q network,

"""

import os
import gym
from UniversalNetwork.universal_DQN import UDQN, myThread
import cv2
import numpy as np
import time


import os

# # 根据不严谨预测，此网页 http://gym.openai.com/envs/#atari 上所以名称中无ram的模型游戏都可以运行
# # 但是各个模型reward所需要的处理大概率不一致，还未修改

# env = gym.make('SpaceInvaders-v0')
# env = gym.make('BreakoutDeterministic-v4')
#env = gym.make('Assault-v0')
# env = gym.make('Alien-v0')
env=gym.make('MsPacman-v0')
env = env.unwrapped


# print('env.action_space',env.action_space)
# print('env.observation_space',env.observation_space)
# print('env.observation_space.shape',env.observation_space.shape)
# print('env.observation_space.high',env.observation_space.high)
# print('env.observation_space.low',env.observation_space.low)
# print('env.reward_range',env.reward_range)

inputImageSize = (100, 80, 1)
# inputImageSize[2] = 1

print('env.action_space.n',env.action_space.n)
print(type(env.action_space.n))
print('env.observation_space.shape[0]',env.observation_space.shape[0])
print(type(env.observation_space.shape[0]))

RL = UDQN(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  observation_shape=inputImageSize,
                  learning_rate=1.0, epsilon_max=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0001,
                  output_graph=True)

total_steps = 0


thread1 = myThread(1, "Thread-1", 1)
thread1.start()
total_reward_list = []
for i_episode in range(100):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    observation = env.reset()
    # 使用opencv做灰度化处理
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))
    total_reward = 0
    start = time.time()
    total_time = 0
    while True:

        env.render()
        # observation_, reward, done, info = env.step(env.action_space.sample())
        # print(env.action_space.sample())
        # # observation_, reward, done, info = env.step(4)  # 4是发送子弹 2、3分别是左右
        # if reward > 0:
        #     print(reward)
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        print(reward)
        # # 给reward做归一化处理
        # end = time.time()
        # #print(end - start)
        # reward = reward / 200
        # # 使用opencv做灰度化处理
        # observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        # observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
        # # cv2.imshow('obe', observation_)
        #
        # RL.store_transition(observation, action, reward, observation_)
        #
        # total_time += (end - start) / 40000
        # total_reward += reward
        #
        #
        # if total_steps > 1000 and total_steps % 2 == 0 and thread1.learn_flag == 1:
        #     t0 = time.time()
        #     RL.learn()
        #     t1 = time.time()
        #     if total_steps < 1010:
        #         print("学习一次时间：", t1 - t0)
        #
        # if done:
        #     total_reward_list.append(total_reward + total_time)
        #     print('episode: ', i_episode,
        #           'total_reward: ', round(total_reward, 2),
        #           'total_time:',round(total_time, 2),
        #           ' epsilon: ', round(RL.epsilon, 2))
        #     # plot_reward()
        #     print('total reward list:', total_reward_list)
        #     break

        # 给reward做处理
        if reward > 0:
            reward = 1
        elif reward < 0 :
            reward = -1

        # 使用opencv做灰度化处理
        observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
        # cv2.imshow('obe', observation_)

        RL.store_transition(observation, action, reward, observation_)

        total_reward += reward
        if total_steps > 1000 and total_steps % 2 == 0 and thread1.learn_flag == 1:
            t0 = time.time()
            RL.learn()
            t1 = time.time()
            if total_steps < 1010:
                print("学习一次时间：", t1 - t0)
        else:
            time.sleep(0.08)
        if done:
            total_reward_list.append(total_reward)
            print('episode: ', i_episode,
                    'total_reward: ', round(total_reward, 2),
                    ' epsilon: ', round(RL.epsilon, 2))
            # plot_reward()
            print('total reward list:', total_reward_list)
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()


