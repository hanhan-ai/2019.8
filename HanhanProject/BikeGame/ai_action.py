#周智圆 2019.8.23
#AI操作函数
#action[left_arrow,right_arrow,spacebar]:1表示执行，0表示不执行
from win32con import *

from Interaction.keyboard_forgame import *

def game_ai_action(action):
    if action[0]==1:
        key_tap(VK_LEFT)   # do action
    elif action[1]==1:
        key_tap(VK_RIGHT)
    elif action[2]==1:
        key_tap(VK_SPACE)
    elif action[3]==1:
        pass
