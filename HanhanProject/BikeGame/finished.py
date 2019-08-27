import pytesseract
from Interaction.keyboard_forgame import *
import cv2
import numpy as np

#作者：赵子轩；最终修改时间：8.27
# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#作者：赵子轩；最终修改时间：8.27
# 差值感知算法
def dHash(img):
    # 缩放8*8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#作者：赵子轩；最终修改时间：8.27
# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

#作者：赵子轩；最终修改时间：8.26
#传入图片img作为参数，判断图片是否符合要求
#
def get_text(img):
    img0= img.crop((300, 210, 380, 240))
    img0.save('../BikeGame/current.jpg')
    img1=cv2.imread('../BikeGame/current.jpg')
    img_modle = cv2.imread('../BikeGame/continue.jpg')
    hash1 = aHash(img1)
    hash_modle = aHash(img_modle)
    n = cmpHash(hash1, hash_modle)
    print('均值哈希算法相似度：' + str(n))
    if n>5:
        return 0
    else:
        return 1
    # i = cropped
    # cropped = cropped.convert('L')  # 把图片强制转成RGB
    # width = cropped.size[0]  # 长度
    # height = cropped.size[1]  # 宽度
    # for i in range(0, width):  # 遍历所有长度的点
    #     for j in range(0, height):  # 遍历所有宽度的点
    #         data = (img.getpixel((i, j)))  # 打印该图片的所有点
    #         if (data[0] >= 150 and data[1] >= 150 and data[2] >= 150):
    #             cropped.putpixel((i, j), (255))
    #
    # cropped = cropped.convert('RGB')
    # cropped.save('H:\kk\kw.jpg')
    # #model_text=
    # text = pytesseract.image_to_string(cropped)
    # print("text=",text)
    #return text

#周智圆 2019.8.27
#游戏通关处理
def game_finished_handle(img):
    if get_text(img)==1:
        click(333,200)
        time.sleep(1.5)
        click(619,476)
        time.sleep(1.5)
        click(500,286)
        time.sleep(1.5)
        #点击第14关
        click(462, 205)
        time.sleep(5)
    else:
        pass

