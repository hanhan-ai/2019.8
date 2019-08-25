#作者：赵子轩；最终修改时间：8.25
#传入图片img作为参数，输出处理完成后输出一个百分数作为reward
def reward_handle(img):
    img = img.crop((200, 453, 440, 467))  # (left, upper, right, lower)
    x = 0
    y = 0
    width = img.size[0]  # 长度
    height = img.size[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = (img.getpixel((i, j)))  # 打印该图片的所有点

            # print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            # print (data[0])#打印RGBA的r值
            if (data[0] <= 70 and data[1] <= 30 and data[2] <= 15 and data[0] >= 45 and data[1] >= 15 and data[2] >= 0):
                x += 1
            y += 1

    return(1 - (x / y))