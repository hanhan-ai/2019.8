import win32gui
import PyHook3

import pythoncom
import pyautogui

import os
import sys

from pymouse import PyMouse
from pykeyboard import PyKeyboard

import threading
from tkinter import *

import time

from BikeGame.base_action import *
from BikeGame.picture_handle import *

from my_keyboard import *

PATH='H:\kk'#截图储存位置
SCREEN_SHOT_TIME=0.04#截屏间隔时间

#周智圆 2019.8.23
#程序界面函数
def first_window(top):
    stop_button = Button(top, text="点我终止程序", command=sys.exit)
    stop_button.pack()

#周智圆 2019.8.23
# 鼠标左击事件处理函数
clicktime=[0]#记录每次点击事件时间
def StartMouseEvent(event):
    global handle,clickn,clicktime
    clicktime.append(time.time())                #事件发生的时间
    handle=event.Window            #窗口句柄
    print(handle)
    #双击左键，进入游戏窗口
    if clicktime[-1]-clicktime[-2]<0.5:
        # 获取游戏窗口位置
        left, top, right, bottom = win32gui.GetWindowRect(handle)
        # 获取游戏窗口句柄的类名和标题
        title = win32gui.GetWindowText(handle)
        clsname = win32gui.GetClassName(handle)
        print(title, clsname)
        # 取消鼠标钩子
        hm.UnhookMouse()
        #开始截屏qs
        print(left, top, right, bottom)
        timer = threading.Timer(SCREEN_SHOT_TIME, screen_shot,(left, top, right-left, bottom-top, PATH,))
        timer.start()
        #模拟游戏输入
        game_base_action()

        # 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截
    return True

#周智圆 2019.8.23
#监听鼠标左击事件函数
def ListenClick():
    global hm
    # 创建HookManager对象
    hm = PyHook3.HookManager()
    # 监听所有鼠标事件
    hm.MouseLeftDown = StartMouseEvent
    # 设置鼠标“钩子”
    hm.HookMouse()
    # 进入循环监听状态
    pythoncom.PumpMessages()

#赵子轩 周智圆 2019.8.23
#截屏函数
picture_i=0
def screen_shot(x,y,w,h,path):
    global picture_i
    picture_i = str(picture_i)
    string1 = 'screenshot'
    string2 = '.png'
    string3 = string1 + picture_i
    string = string3 + string2
    print(string)
    img = pyautogui.screenshot(region=[x,y,w,h])
    img=pic_change(img)#图片预处理
    img.save(os.path.join(path,os.path.basename(string)))
    picture_i = int(picture_i)
    picture_i +=1
    global timer
    timer = threading.Timer(SCREEN_SHOT_TIME, screen_shot,(x,y,w,h,path,))
    timer.start()

#周智圆 2019.8.23
#main函数
if __name__ == "__main__":
    top=Tk()
    first_window(top)
    t=threading.Thread(target=ListenClick,)
    t.setDaemon(True)#设为守护线程
    t.start()
    top.mainloop()




