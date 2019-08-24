from Interaction.my_keyboard import *
import threading
import time
#周智圆 2019.8.23
#游戏默认操作函数
def fun():
    key_press('up_arrow')
    time.sleep(None)
    key_up('up_arrow')

#周智圆 2019.8.23
#启动游戏默认操作线程操作函数
def game_base_action():
    t = threading.Thread(target=fun, )
    t.setDaemon(True)  # 设为守护线程
    t.start()
