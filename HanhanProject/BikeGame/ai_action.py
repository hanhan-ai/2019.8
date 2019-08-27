#周智圆 2019.8.23
#AI操作函数
#action[left,right,spacebar,]:1表示执行，0表示不执行
from win32con import *
from Interaction import global_var_model as gl
from Interaction.keyboard_forgame import *

def game_ai_action(action):
    if action[0]==1:
        print(gl.HANDLE,"执行了0000000000000000000000000000000000000000000000000000000000")
        key_tap(VK_LEFT)   # do action
    elif action[1]==1:
        print(gl.HANDLE,"执行了111111111111111111111111111111111111111111111111111111111")
        key_tap(VK_RIGHT)
    elif action[2]==1:
        print(gl.HANDLE,"执行了222222222222222222222222222222222222222222222222222222222")
        key_tap(VK_SPACE)
    elif action[3]==1:
        print(gl.HANDLE,"执行了333333333333333333333333333333333333333333333333333333333")
        pass
