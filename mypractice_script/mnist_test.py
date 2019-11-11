import numpy as np
from random import randint


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    temp_shape = x.shape
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    grad = grad.flatten()
    x = x.flatten()
    for indx in range(x.size):
        tmp_val = x[indx]
        # f(x+h)的计算
        x[indx] = tmp_val + h
        x = x.reshape(temp_shape)
        fxh1 = f(x)
        x = x.flatten()
        # f(x-h)的计算
        x[indx] = tmp_val - h
        x = x.reshape(temp_shape)
        fxh2 = f(x)
        x = x.flatten()
        grad[indx] = (fxh1 - fxh2) / (2*h)
        x[indx] = tmp_val # 还原值
    grad = grad.reshape(temp_shape)
    x = x.reshape(temp_shape)
    return grad

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        self.para = {}  #字典
        self.para['W1'] = 0.01* np.random.randn(input_size,hidden_size)
        self.para['B1'] = 0.01* np.random.randn(1,hidden_size)
        self.para['W2'] = 0.01* np.random.randn(hidden_size,output_size)
        self.para['B2'] = 0.01* np.random.randn(1,output_size)         #参数

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def relu(self,x):
        return np.maximum(0, x)

    def softmax(self,a):
        c = np.max(a,axis=1)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    def forward(self,input):
        A1 = np.dot(input,self.para['W1'])+self.para['B1']
        Z1 = self.relu(A1)
        A2 = np.dot(Z1,self.para['W2'])+self.para['B2']
        Y = self.softmax(A2)
        return Y

    def mean_squared_error(self,y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(self,y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def numerical_diff(self,f,x):
        h = 1e-4
        return (f(x+h) - f(x-h))/(2*h)

    # def numerical_gradient(self,f,x):
    #     out = np.zeros_like(x,float)
    #     h = 1e-4
    #     for idx in range(x.size):
    #         tmp = np.zeros_like(x,float)
    #         tmp[idx] = h
    #         f_x_add_h = f(x+tmp)
    #         f_x_minus_h = f(x-tmp)
    #         out[idx] = (f_x_add_h-f_x_minus_h)/(2*h)
    #     return out

    def loss(self,input,t):
        y = self.forward(input) #获取输入前向传播的输出
        return (self.cross_entropy_error(y,t))

    def accuracy(self,input,t):
        y = self.forward(input)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,input,t):
        loss_w = lambda w: self.loss(input,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_w,self.para['W1'])
        grads['B1'] = numerical_gradient(loss_w, self.para['B1'])
        grads['W2'] = numerical_gradient(loss_w, self.para['W2'])
        grads['B2'] = numerical_gradient(loss_w, self.para['B2'])
        return grads


#test

# net = TwoLayerNet(2,4,2)
# input = np.array([[2,4]])
# print(input)
# print(input.shape)
# print(net.W1)
# print(net.W1.shape)
# print(net.B1)
# print(net.B1.shape)
# print(net.W2)
# print(net.W2.shape)
# print(net.B2)
# print(net.B2.shape)
# print(np.dot(input,net.W1)+net.B1)
# ss = np.dot(input,net.W1)+net.B1
# print(ss.shape)
# print(net.forward(input))

#
# t = np.random.randint(0,10,10)
# tt = np.random.randn(10)
#
# print(type(tt))
# print(tt)
# print(t)

net = TwoLayerNet(3,3,3)
print(net.para['W1'].shape)
input = np.random.rand(1,3)
t = np.random.rand(1,3)
y = net.forward(input)
grads = net.numerical_gradient(input,t)
print(grads)

