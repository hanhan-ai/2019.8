"""
Deep Q network,

"""

import os
import gym
from universal_DQN import UDQN, init_model,init_UDQN
import cv2
import numpy as np
import time
import os

# 根据不严谨预测，此网页 http://gym.openai.com/envs/#atari 上所以名称中无ram的模型游戏都可以运行
# 但是各个模型reward所需要的处理大概率不一致，还未修改
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('BreakoutDeterministic-v4')
# env = gym.make('Assault-v0')
# env = gym.make('Asterix-v0')
# env = gym.make('Alien-v0')
# env = gym.make('MsPacman-v0')
# env = env.unwrapped
# print('env.action_space',env.action_space)
# print('env.observation_space',env.observation_space)
# print('env.observation_space.shape',env.observation_space.shape)
# print('env.observation_space.high',env.observation_space.high)
# print('env.observation_space.low',env.observation_space.low)
# print('env.reward_range',env.reward_range)
# inputImageSize = (100, 80, 1)
# inputImageSize[2] = 1
# print('env.action_space.n',env.action_space.n)
# print(type(env.action_space.n))
# print('env.observation_space.shape[0]',env.observation_space.shape[0])
# print(type(env.observation_space.shape[0]))
# RL = UDQN(n_actions=env.action_space.n,
#                   n_features=env.observation_space.shape[0],
#                   observation_shape=inputImageSize,
#                   learning_rate=1.0, epsilon_max=0.9,
#                   replace_target_iter=100, memory_size=2000,
#                   e_greedy_increment=0.0001,
#                   output_graph=True)
# total_steps = 0
# thread1 = myThread(1, "Thread-1", 1)
# thread1.start()
# total_reward_list = []

# env = init_model('SpaceInvaders-v0')
env = init_model('MsPacman-v0')
# print('env    ',type(env))
# print(env)
# print('env.action_space',env.action_space)
# print('env.observation_space.shape[0]',env.observation_space.shape[0])

# 选择Adadelta优化器，输入学习率参数
# lr=1.0, rho=0.95, epsilon=None, decay=0.0
# 选择RMSprop优化器，输入学习率参数
# lr=0.001, rho=0.9, epsilon=None, decay=0.0
# 选择Adam优化器，输入学习率参数
# lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
# 选择Nadam优化器，输入学习率参数
# lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004
#
# init_UDQN参数列表env, inputImageSize, choose_optimizers ,lr, rho, beta_1, beta_2, decay, amsgrad, schedule_decay
# ,0.95,0.0,0.0,1e-6,False,0.0
RL = init_UDQN(env,(100, 80, 1),'Adadelta',1.0)

for i_episode in range(10):

    # run()参数列表env, inputImageSize, total_steps, total_reward_list, i_episode
    RL.run(env,(100,80,1),0,[],i_episode,1)

    # #  重置游戏
    # observation = env.reset()
    #
    # # 使用opencv做灰度化处理
    # observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))
    #
    # total_reward = 0
    # start = time.time()
    # total_time = 0
    #
    # while True:
    #
    #     # 重新绘制一帧
    #     env.render()
    #
    #     action = RL.choose_action(observation)
    #     # observation_  用于表示游戏的状态
    #     # reward        上一个action获得的奖励
    #     # done          游戏是否结束
    #     # info          用于调试
    #     observation_, reward, done, info = env.step(action)
    #
    #     # # 给reward做处理，SpaceInvaders-v0使用
    #     # end = time.time()
    #     # #print(end - start)
    #     # reward = reward / 200
    #     # # 使用opencv做灰度化处理
    #     # observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
    #     # observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
    #     # # cv2.imshow('obe', observation_)
    #     #
    #     # RL.store_transition(observation, action, reward, observation_)
    #     #
    #     # total_time += (end - start) / 40000
    #     # total_reward += reward
    #     #
    #     #
    #     # if total_steps > 1000 and total_steps % 2 == 0 and thread1.learn_flag == 1:
    #     #     t0 = time.time()
    #     #     RL.learn()
    #     #     t1 = time.time()
    #     #     if total_steps < 1010:
    #     #         print("学习一次时间：", t1 - t0)
    #     #
    #     # if done:
    #     #     total_reward_list.append(total_reward + total_time)
    #     #     print('episode: ', i_episode,
    #     #           'total_reward: ', round(total_reward, 2),
    #     #           'total_time:',round(total_time, 2),
    #     #           ' epsilon: ', round(RL.epsilon, 2))
    #     #     # plot_reward()
    #     #     print('total reward list:', total_reward_list)
    #     #     break
    #
    #     # # 给reward做处理,BreakoutDeterministic-v4使用
    #     # if reward > 0:
    #     #     reward = 1
    #     # elif reward < 0:
    #     #     reward = -1
    #
    #     #用于MsPacman-v0模型，未必最优
    #     reward = reward / 10
    #     # if reward>0:
    #     #     print('reward     ',reward)
    #
    #     # 使用opencv做灰度化处理
    #     observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
    #     observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
    #
    #     RL.store_transition(observation, action, reward, observation_)
    #
    #     total_reward += reward
    #     # if total_steps > 1000 and total_steps % 2 == 0 and thread1.learn_flag == 1:
    #     #     t0 = time.time()
    #     #     RL.learn()
    #     #     t1 = time.time()
    #     #     if total_steps < 1010:
    #     #         print("学习一次时间：", t1 - t0)
    #     # else:
    #     #     time.sleep(0.08)
    #     if done:
    #         total_reward_list.append(total_reward)
    #         print('episode: ', i_episode,
    #                 'total_reward: ', round(total_reward, 2),
    #                 ' epsilon: ', round(RL.epsilon, 2))
    #         # plot_reward()
    #         print('total reward list:', total_reward_list)
    #         break
    #
    #     observation = observation_
    #     total_steps += 1

RL.plot_cost()


