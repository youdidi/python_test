import numpy as np
# x = np.array([1.0, 2.0, 3.0])
# print(x)
# print(type(x))
# y = np.array([2.0, 4.0, 6.0])
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
# print(x/2)
#
# A = np.array([[1, 2], [3, 4]])
# print(A)
# print(A.shape)
# print(A.dtype)
# B = np.array([[3, 0], [0, 6]])
# print(A+B)
# print(A*B)

#

import matplotlib.pyplot as plt
from matplotlib.image import imread
#
# x = np.arange(0, 6, 0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)
# print(x)
# print(y1)
# print(y2)
# plt.plot(x, y1, label="sin")
# plt.plot(x, y2, linestyle="--", label="cos")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title('sin & cos')
# plt.legend()
# plt.show()

# img = imread('test.jpg')
# plt.imshow(img)
# plt.show()

# x = np.array([0, 1]) #input
# w = np.array([0.5, 0.5]) #weight
# b = -0.7
# print(w*x)
# print(np.sum(w*x))
# print(np.sum(w*x)+b)
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) #weight
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <=0:
        return 0
    else:
        return 1

def step_function(x):
    return np.array(x >0, dtype=np.int)

# x = np.arange(-5.0, 5.0 ,0.1)
# y = step_function(x)
# plt.plot(x,y)
# # plt.ylim(-0.1, 1.1)
# plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x
# x = np.arange(-5.0, 5.0 ,0.1)
# y = relu(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()
# A = np.array([[1,2],[3,4]])
# B = np.array([[1,0],[0,1]])
# print(np.ndim(B))
# print(B.shape)
# print(np.ndim(A))
# print(A.shape)
# print(np.dot(A, B))
# print(A*B)

# X = np.array([1.0, 0.5])
# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])
# print(X.shape)
# print(W1.shape)
# print(B1.shape)
# A1 = np.dot(X,W1) + B1
# print(A1)
# Z1 = sigmoid(A1)
# print(Z1)
#
# W2 = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
# B2 = np.array([0.1, 0.2])
#
# A2 = np.dot(Z1,W2) + B2
# Z2 = sigmoid(A2)
#
# W3 = np.array([[0.1, 0.3],[0.2 , 0.4]])
# B3 = np.array([0.1, 0.2])
#
# A3 = np.dot(Z2, W3) + B3
#
# Y = identity_function(A3)
#
# print(Y)

# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])
#     return network
#
# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = identity_function(a3)
#     return y
#
# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y) # [ 0.31682708 0.69627909]

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# a = np.array([0.3, 2.9, 4.0])
# # exp_a = np.exp(a)
# # print(exp_a)
# #
# # sum_exp_a = np.sum(exp_a)
# # print(sum_exp_a)
# # y = exp_a/sum_exp_a
# # print(y)

# a = np.array([-1010,-1000,-900])
# print(np.exp(a)/np.sum(np.exp(a)))

# a = np.array([0.3, 2.9, 4.0])
# y = softmax(a)
# print(y)
# print(np.sum(y))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# print(np.array(y)-np.array(t))
# print((np.array(y)-np.array(t))**2)
# print(mean_squared_error(np.array(y),np.array(t)))

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

# def numerical_gradient(f,x):
#     out = np.zeros_like(x,float)
#     h = 1e-4
#     for idx in range(x.size):
#         tmp = np.zeros_like(x,float)
#         tmp[idx] = h
#         f_x_add_h = f(x+tmp)
#         f_x_minus_h = f(x-tmp)
#         out[idx] = (f_x_add_h-f_x_minus_h)/(2*h)
#     return out

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


def function_2(x):
    return np.sum(x**2)

def gradient_descent(f,init_x,lr=0.01,step_num=300):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x = x-lr*grad
    return x

print(numerical_gradient(function_2,np.array([[[3.,0.],[2.,2.]],[[2.,4.],[1.,3.]]])))
#
# print(gradient_descent(function_2,np.array([2.,2.])))
#
#
# print(numerical_diff(function_1,3))

# a = np.arange(10)
# a = a.reshape(5,2)
# print(a)
# indices = np.array([1,3])
# indices = np.where(a<5)
# indices2 = (np.array([1,2,3,4,5,6]),)
# print(indices)
# print(a[indices])
# print(a[indices2])
# print(type(indices))
# s = slice(1,5)
# print(s)
# print(a[s])
# print(a[2:7:2])
# a = np.arange(9)
# a= a.reshape(3,3)
# mask = np.ones_like(a)
# indices = np.where(mask)
# print(indices)
# print(mask)
# b = a[indices]
# print()
# print(type(b))


# print(a[indices])
# indices = np.where()
# print(indices)