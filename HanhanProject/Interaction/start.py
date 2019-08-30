import ctypes
import inspect

import win32gui
import win32api

import PyHook3
import pythoncom
import pyautogui

import os
import sys

from pymouse import PyMouse
# from pykeyboard import PyKeyboard

import threading
from tkinter import *
import signal

import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from BikeGame.ai_action import *
from BikeGame.base_action import *
from BikeGame.finished import *

from BikeGame.picture_handle import *
from BikeGame.reward_handle import *
from Interaction import global_var_model as gl


PATH='H:\kk'#截图储存位置
SCREEN_SHOT_TIME=0.04#截屏间隔时间


#周智圆 2019.8.27
#点击按钮退出程序
def stop_event():
    print(gl.HANDLE)
    click(10, 20)
    gl.STATE = False
    time.sleep(0.6)

    print("finally")
    print('down,up',gl.down,gl.up)
    gl.HANDLE = -1
    print("==stop...")
    while True:
        if gl.RELEASE ==True:
            sys.exit()
#周智圆 2019.8.23
#程序界面函数
def first_window(top):
    print("==============")
    stop_button = Button(top, text="点我终止程序", command=stop_event)
    stop_button.pack()
from DeepQNetworkBall.Network import *

#周智圆 2019.8.23
# 鼠标左击事件处理函数
clicktime=[0]#记录每次点击事件时间
def StartMouseEvent(event):
    print("srart....")
    global clickn,clicktime
    clicktime.append(time.time())                #事件发生的时间
    handle=event.Window #窗口句柄
    print(handle)
    print('鼠标点击坐标',event.Position)
    #双击左键，开始，进入游戏窗口
    if clicktime[-1]-clicktime[-2]<0.5 and handle!=gl.HANDLE:
        gl.HANDLE=event.Window            #游戏窗口句柄
        # 获取游戏窗口位置
        gl.LEFT, gl.TOP, gl.RIGHT, gl.BOTTOM \
            = win32gui.GetWindowRect(gl.HANDLE)
        # 获取游戏窗口句柄的类名和标题
        title = win32gui.GetWindowText(gl.HANDLE)
        clsname = win32gui.GetClassName(gl.HANDLE)
        print(title, clsname)
        print("游戏窗口",gl.LEFT, gl.TOP, gl.RIGHT, gl.BOTTOM)
        # 取消鼠标钩子
        hm.UnhookMouse()
        #记录游戏开始时间
        gl.STARTETIME=time.time()
        #模拟游戏输入
        game_base_action()
        #开始进行神经网络的循环
        #赵士陆 2019.8.25  20：54
        startNetwork()
        # 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截
    # 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截
    return True

#周智圆 2019.8.23
#监听鼠标左击事件函数
def ListenClick():
    global hm
    # 创建HookManager对象
    hm = PyHook3.HookManager()
    # 监听所有鼠标左击事件
    hm.MouseLeftDown = StartMouseEvent
    # 设置鼠标“钩子”
    hm.HookMouse()
    # 进入循环监听状态
    pythoncom.PumpMessages()

#周智圆 2019.8.25
#通用转化函数函数
#acton:游戏操作
#rw:reward  frame:环境状态数组
def game_convertion(action):
    game_ai_action(action)
    img = pyautogui.screenshot(region=[gl.LEFT, gl.TOP, gl.RIGHT - gl.LEFT, gl.BOTTOM - gl.TOP])
    print(img)
    imgfi = img
    imgr = img
    imgfr = img
    game_finished_handle(imgfi)
    rw=reward_handle(imgr)
    frame=pic_change(imgfr)
    return rw,frame

'''
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
'''

#周智圆 2019.8.23
#main函数
if __name__ == "__main__":
    TOP=Tk()
    stop_button = Button(TOP, text="点我终止程序", command=stop_event)
    stop_button.pack()
    t=threading.Thread(target=ListenClick,)
    t.setDaemon(True)#设为守护线程
    t.start()
    TOP.mainloop()







