from PIL import Image
from numpy import *
#赵子轩 2019.8.23
#截屏图片处理函数
#img 处理图片
def picture_handle(img):
    width = img.size[0]  # 长度
    height = img.size[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = (img.getpixel((i, j)))  # 打印该图片的所有点
            # print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            # print (data[0])#打印RGBA的r值
            if (data[0] >= 120 and data[1] >= 120 and data[2] >= 120 and data[0] <= 160 and data[1] <= 160 and data[
                2] <= 160):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
                img.putpixel((i, j), (255, 0, 0, 255))  # 红色,石头
            if (data[0] >= 150 and data[1] >= 130 and data[2] <= 30):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
                img.putpixel((i, j), (255, 255, 0, 255))  # 黄色，金块
            if (data[0] >= 150 and data[1] >= 150 and data[2] >= 150):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
                img.putpixel((i, j), (255, 255, 255, 255))  # 白色，钻石
            # if (data[0] <= 220 and data[1] <= 180 and data[1] >= 50):  # RGBA的r值大于170，并且g值大于170,并且b值大于170
            #     img.putpixel((i, j), (0, 0, 0, 255))  # 黑色，背景
    img = img.convert("RGB")  # 把图片强制转成RGB

    imagearray = array(img)
    print(imagearray)

    for i in range(1, 5):
        imagearray = 255.0 * (imagearray / 255.0) ** 2
        i += 1

    img = Image.fromarray(uint8(imagearray))
    img = img.convert("L")
    img.save('return golden achieve5.jpg')  # 保存修改像素点后的图片
