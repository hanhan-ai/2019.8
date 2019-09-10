# ======================================================================================================================
# ===========================================Game Handle Part===========================================================
# ======================================================================================================================
"""
作者：赵士陆
创建时间：2019.9.7
最后一次修改时间：2019.9.10

"""
# An example game to test the GA & Network
from GApart.Brain import *
from GApart.population import *
import gym
import numpy as np

OBSERVE_NUM = 4
HIDDENLAYER_1 = 4
HIDDENLAYER_2 = 5
ACTION = 2

env = gym.make('CartPole-v0')
population = Population(10)
population.initPopulation(OBSERVE_NUM, HIDDENLAYER_1, HIDDENLAYER_2, ACTION)

population.runGame(env)