from mytest.two_net import *
import numpy as np
if __name__ == '__main__':
    #本测试用于测试全连接对部分函数（例如：指数函数等）的拟合效果。
    #首先准备一个函数数据
    input_data = np.arange(-5,5,0.01)
    out_data_right = np.exp(input_data)

    print(input_data.size)
    print(out_data_right.size)


    #再初始化一个训练网络
    two_net_fun = two_net(lay1_size=(1,1),lay2_size=(1,1),action_fun=0,loss_fun=0,batch_size=1000,learn_rate=0.1)
