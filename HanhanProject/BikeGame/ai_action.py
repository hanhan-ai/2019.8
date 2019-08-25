#周智圆 2019.8.23
#AI操作函数
#action[left_arrow,right_arrow,spacebar]:1表示执行，0表示不执行

from Interaction.my_keyboard import *

def game_ai_action(action):
    if action[0]==1:
        key_tap('left_arrow')   # do action
    elif action[1]==1:
        key_tap('right_arrow')
    elif action[2]==1:
        key_tap('spacebar')
    elif action[3]==1:
        pass
