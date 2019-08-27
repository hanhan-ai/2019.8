from win32con import *

from Interaction.keyboard_forgame import *
import threading
import time

#周智圆 2019.8.23
#游戏默认操作函数
def fun():
    try:
        key_press(VK_UP)
        print("====go...")
        time.sleep(10000)
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