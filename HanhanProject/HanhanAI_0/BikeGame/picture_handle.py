from PIL import Image
from numpy import *
import numpy as np
from Interaction import global_var_model as gl

#作者：赵子轩；最终修改时间：8.25
#传入图片img作为参数，输出处理完成的图片对应的数组
def pic_change(img):
    width = img.size[0]  # 长度
    height = img.size[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = (img.getpixel((i, j)))  # 打印该图片的所有点
            if (data[0] <= 20 and data[1] <= 20 and data[2] <= 20):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
                img.putpixel((i, j), (255, 0, 0, 255))  #
            elif (data[0] >= 115 and data[1] >= 75 and data[2] >= 0 and data[0] <= 170 and data[1] <= 140 and data[
                2] <= 80):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
                img.putpixel((i, j), (0, 0, 255, 255))  #
            else:
                img.putpixel((i, j), (0, 255, 0, 255))  #

            # print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            # print (data[0])#打印RGBA的r值
            # if (data[0] <= 10 and data[1] <= 10 and data[2] <= 18):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
            #     img.putpixel((i, j), (255, 0, 0, 255))  #
            # if (data[0] >= 115 and data[1] >= 120 and data[2] >= 120 and data[0] <= 170 and data[1] <= 180 and data[
            #     2] <= 160):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
            #     img.putpixel((i, j), (255, 255, 255, 255))  #
            # if (data[0] >= 75 and data[1] >= 30 and data[0] <= 110 and data[1] <= 60 and data[
            #     2] <= 20):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
            #     img.putpixel((i, j), (255, 255, 255, 255))  #
            #if (data[0] <= 30 and data[1] <= 30 and data[2] <= 30):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
            #   img.putpixel((i, j), (255, 255, 255, 255))  #
    img = img.convert("L")
    img = img.convert("RGB")

    width = img.size[0]  # 长度
    height = img.size[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = (img.getpixel((i, j)))  # 打印该图片的所有点
            # print('data',data[0],data[1],data[2])
            if (data[0] <= 100 and data[1] <= 100 and data[2] <= 100):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
                img.putpixel((i, j - 1), (0, 0, 0, 255))  #
                img.putpixel((i, j - 2), (0, 0, 0, 255))  # img = img.convert("L")
    img = img.convert("L")
    #img.save('../BikeGame/jietu.jpg')
    x,y=img.size
    img=img.crop((0, 0,x,y ))
    imgnp = np.array(img)
    return imgnp  # (left, upper, right, lower)


