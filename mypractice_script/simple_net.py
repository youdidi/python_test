import numpy as np

class simplenet:
    def __init__(self):
        # self.W = np.random.randn(2,3)
        self.W = np.array([[0.5,0.6,0.3],[0.4,0.7,0.5]])

    def softmax(self,a):
        c = np.max(a,axis=1)
        numtmp = c.size
        c = c.reshape(numtmp,1)
        b = a-c
        exp_a = np.exp(b)
        sum_exp_a = np.sum(exp_a,axis=1)
        numtmp2 = sum_exp_a.size
        sum_exp_a = sum_exp_a.reshape(numtmp2,1)
        y = exp_a / sum_exp_a
        return y

    def cross_entropy_error(self,y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def predict(self,x):
        z = np.dot(x,self.W)
        y = self.softmax(z)
        return y

    def loss(self,x,t):
        y = self.predict(x)
        loss = self.cross_entropy_error(y,t)
        return loss
    def numerical_gradient(self):
        h = 1e-4 # 0.0001
        x = self.W
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




# test





# net = simplenet()
# x = np.array([[2,4],[3,2],[7,13],[23,45]])
# t = np.array([[0,0,1],[1,0,0],[1,0,0],[0,1,0]])
# # x = np.array([[2,4]])
# # t = np.array([[0,0,1]])
# print(net.loss(x,t))
# # print(net.predict(x))
#
# # x = np.array([[3,2]])
# x = np.array([[2,4]])
# t = np.array([[0,0,1]])
# print(net.loss(x,t))
#





















# net = simplenet()
# net.W = np.array([[ 0.47355232, 0.9977393, 0.84668094],[ 0.85557411, 0.03563661, 0.69422093]])
# # print(net.W)
# x = np.array([0.6, 0.9])
# # p = net.predict(x)
# # print(p)
# # print(np.argmax(p))
# t = np.array([0, 0, 1])
# print(net.loss(x, t))














# def f(W):
#     net.W = W
#     return net.loss(x,t)
#
# def numerical_gradient(f, x):
#     h = 1e-4 # 0.0001
#     temp_shape = x.shape
#     grad = np.zeros_like(x) # 生成和x形状相同的数组
#     grad = grad.flatten()
#     x = x.flatten()
#     for indx in range(x.size):
#         tmp_val = x[indx]
#         # f(x+h)的计算
#         x[indx] = tmp_val + h
#         x = x.reshape(temp_shape)
#         fxh1 = f(x)
#         x = x.flatten()
#         # f(x-h)的计算
#         x[indx] = tmp_val - h
#         x = x.reshape(temp_shape)
#         fxh2 = f(x)
#         x = x.flatten()
#         grad[indx] = (fxh1 - fxh2) / (2*h)
#         x[indx] = tmp_val # 还原值
#     grad = grad.reshape(temp_shape)
#     x = x.reshape(temp_shape)
#     return grad
#
# dw = numerical_gradient(f,net.W)
# print(dw)