'''
创建人：赵子轩
创建时间：2019.8.23
最后一次修改时间：2019.8.25
'''
from HanhanAI_0.Interaction import global_var_model as gl
#作者：赵子轩；最终修改时间：8.25
#传入图片img，传入LAST_ADR上一次处理值，输出一个0或1作为reward
def reward_handle(img):
    left=int(329/998*(gl.RIGHT-gl.LEFT))
    upper=int(707/749*(gl.BOTTOM-gl.TOP))
    right=int(685/998*(gl.RIGHT-gl.LEFT))
    lower=int(731/749*(gl.BOTTOM-gl.TOP))
    print(left, upper, right, lower)
    img1 = img.crop((left, upper, right, lower ))  # (left, upper, right, lower)
    #img1.save('../BikeGame/jindutiao.jpg')
    x = 0
    y = 0
    width = img1.size[0]  # 长度
    height = img1.size[1]  # 宽度
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = (img1.getpixel((i, j)))  # 打印该图片的所有点
            # print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            # print (data[0])#打印RGBA的r值
            if (data[0] <= 70 and data[1] <= 30 and data[2] <= 15 and data[0] >= 45 and data[1] >= 15 and data[2] >= 0):
                x += 1
            y += 1
    print('进度条之前：',gl.LAST_ADR,'进度条现在：',1-(x/y))
    if ((1-(x/y))<(gl.LAST_ADR)*0.5) :  #肾上腺素减少到之前的50%
        gl.LAST_ADR = 1 - (x / y)
        return -1
    elif x==0:
        return -1
    else:
        gl.LAST_ADR = 1 - (x / y)
        return 1
