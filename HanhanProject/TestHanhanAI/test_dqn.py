#测试DQN
from HanhanAI.universal_dqn import *
import gym

env = init_model('SpaceInvaders-v0')
RL = init_UDQN(env,(100, 80, 1),'Adam',1.0)
RL.run(env,(100,80,1),0,[],100,50)
RL.plot_cost()


