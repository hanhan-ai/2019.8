# ======================================================================================================================
# ===========================================Game Handle Part===========================================================
# ======================================================================================================================

from GApart.Brain import *
from GApart.population import *
import gym
import numpy as np

OBSERVE_NUM = 4
HIDDENLAYER_1 = 4
HIDDENLAYER_2 = 5
ACTION = 2

env = gym.make('CartPole-v0')
population = Population(3)
population.initPopulation(OBSERVE_NUM, HIDDENLAYER_1, HIDDENLAYER_2, ACTION)

population.runGame(env)