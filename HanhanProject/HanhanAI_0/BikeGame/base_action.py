'''
创建人：周智圆
创建时间：2019.8.21
最后一次修改时间：2019.8.23
'''

from win32con import *
from HanhanAI_0.Interaction.keyboard_forgame import *

import threading
from HanhanAI_0.Interaction import global_var_model as gl

#周智圆 2019.8.23
#游戏默认操作函数
def fun():
    try:
        while gl.STATE==True:
            key_tap(VK_UP)
    except Exception as e:
        print(e)
    finally:
        pass


#周智圆 2019.8.23
#启动游戏默认操作线程操作函数
def game_base_action():
    t = threading.Thread(target=fun, )
    t.setDaemon(True)  # 设为守护线程
    t.start()