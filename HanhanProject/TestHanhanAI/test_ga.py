#测试包含遗传算法的神经网络
from HanhanAI.population import *
import gym

OBSERVE_NUM = 4
HIDDENLAYER_1 = 4
HIDDENLAYER_2 = 5
ACTION = 2

env = gym.make('CartPole-v0')
population = Population(3)
population.initPopulation(OBSERVE_NUM, HIDDENLAYER_1, HIDDENLAYER_2, ACTION)

population.runGame(env)