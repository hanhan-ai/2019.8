# ======================================================================================================================
# ===========================================Game Handle Part===========================================================
# ======================================================================================================================

from GeneticAlgorithmPart.Population import *
import gym
import numpy as np

HIDDENLAYER_1 = 10
HIDDENLAYER_2 = 6
ACTION = 2


env = gym.make('CartPole-v0')
observation = env.reset()

population = Population(30)
population.initPopulation(len(observation), HIDDENLAYER_1, HIDDENLAYER_2, ACTION)

while True:

    print('-'*20, 'Generation ', population.biont[0].generation, ' ', '-'*20)

    # population.loadNet()
    for i in range(population.size_population):
        print('======== Biont ', i, ' Playing... ========')
        reward_vec = []
        for j in range(10):
            episode_reward = 0
            while True:
                env.render()  # 更新并渲染游戏画面
                action_vec = population.biont[i].run(observation)
                action = np.argmax(action_vec)
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    env.reset()
                    reward_vec.append(episode_reward)
                    print('Episode ', j, ' | score = ', episode_reward)
                    break
        population.biont[i].evaluate_score = np.mean(reward_vec)
        print('Biont ', i, ' evaluate score is: ', population.biont[i].evaluate_score)

    father, mother = population.selectParents()
    population.breed(father, mother)

    population.saveNet(population.biont[0].generation)