#作者：赵子轩；最终修改时间：8.26
#传入图片img作为参数，输出处理完成图片中的文字
#识别出CONTINUE时返回fel
import pytesseract
from PIL import Image
def get_text(img):
    if __name__ == '__main__':
        cropped = img.crop((300, 210, 380, 240))
        cropped = cropped.convert('L')  # 把图片强制转成RGB
        width = cropped.size[0]  # 长度
        height = cropped.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (img.getpixel((i, j)))  # 打印该图片的所有点
                if (data[0] >= 150 and data[1] >= 150 and data[2] >= 150):
                    cropped.putpixel((i, j), (255, 255, 255, 255))

        cropped = cropped.convert('RGB')
        #cropped.save('wz1 return.jpg')

        text = pytesseract.image_to_string(cropped)
        return text
