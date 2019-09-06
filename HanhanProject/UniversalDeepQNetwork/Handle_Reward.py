import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # x = np.arange(0, y数组中元素数量, 1)
    # y = np.array(多个reward)
    # 下方为实例
    x = np.arange(0, 100, 1)
    y = np.array([4.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 3.0, 3.0, 2.0, 0.0, 0.0, 1.0, 3.0, 3.0, 0.0, 2.0, 3.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 4.0, 1.0, 2.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 4.0, 1.0, 0.0]
                 )
    z1 = np.polyfit(x, y, 1)  # 用3次多项式拟合

    print('斜率*100w = ',z1[0]*1000000)

    p1 = np.poly1d(z1)
    print('p1',p1)  # 在屏幕上打印拟合多项式
    yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
    plt.title('polyfitting')
    plt.show()
    plt.savefig('p1.png')


