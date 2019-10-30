import numpy as np


class layer:
    # 定义基本属性
    W_matrix = 0
    B_matrix = 0
    A_matrix = 0
    Z_matrix = 0
    X_matrix = 0
    DFDZ_matrix = 0
    DFDA_matrix = 0
    DFDW_matrix = 0
    DFDB_matrix = 0
    DFDX_matrix = 0
    DFDA_matrix = 0
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __input_size = 0
    __output_size = 0
    __active_fun = 0      #active fun type ; sigmod函数:0 ;  relu函数:1;

    # 定义构造方法
    def __init__(self,input_size,output_size,active_fun):
        """
           example:  输入10个4元的向量，输出为10个3元的数据
           lay = layer(input_size=(10,4),output_size=(10,3),active_fun=0)
           """
        self.__active_fun = active_fun
        self.__input_size = input_size
        self.__output_size = output_size
        self.W_matrix = np.random.randn(input_size[1],output_size[1])  #随机初始化W矩阵，权重矩阵
        self.B_matrix = np.random.randn(1,output_size[1])     #随机初始化B矩阵，偏置矩阵
        # self.Z_matrix = np.zeros(shape=(1,output_size),dtype=float)     #初始化Z矩阵为全0矩阵
        # self.A_matrix = np.zeros(shape=(1,output_size),dtype=float)     #初始化A矩阵为全0矩阵
        # self.DFDZ_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDZ_matrix矩阵为全0矩阵
        # self.DFDA_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDA_matrix矩阵为全0矩阵
        # self.DFDW_matrix = np.zeros(shape=(input_size, output_size), dtype=float)  # 初始化DFDW_matrix矩阵为全0矩阵
        # self.DFDB_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDB_matrix矩阵为全0矩阵
        # self.DFDX_matrix = np.zeros(shape=(1, input_size), dtype=float)  # 初始化DFDX_matrix矩阵为全0矩阵
        # self.DFDA_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDA_matrix矩阵为全0矩阵

    def set_X_matrix(self,input_matrix):        #设置输入矩阵
        self.X_matrix = input_matrix
        if self.__input_size != self.X_matrix.shape:
            raise TypeError('bad operand type')
        pass

    def set_DFDA_matrix(self,DFDA_matrix):        #设置反向传输需要的偏导数矩阵
        self.DFDA_matrix = DFDA_matrix
        if self.__output_size != self.DFDA_matrix.shape:
            raise TypeError('bad operand type')
        pass

    def forward(self):                              #根据输入矩阵以及本层参数，计算输出矩阵(计算前向)
        self.Z_matrix = np.dot(self.X_matrix , self.W_matrix) + self.B_matrix     #计算Z矩阵
        if self.__active_fun == 0:      #如果为sigmoid函数
            self.A_matrix = self.sigmoid(self.Z_matrix)
        elif self.__active_fun == 1:    #如果为relu函数
            self.A_matrix = self.relu(self.Z_matrix)
        pass


    def backward(self):                               #根据偏导数矩阵，进行反向传输计算
        if self.__active_fun == 0:  # 如果为sigmoid函数
            DADZ_matrix = self.Dsigmoid(self.Z_matrix)
            self.DFDZ_matrix = self.DFDA_matrix * DADZ_matrix
        elif self.__active_fun == 1:  # 如果为relu函数
            DADZ_matrix = self.Drelu(self.Z_matrix)
            self.DFDZ_matrix = self.DFDA_matrix * DADZ_matrix

        print((self.DFDZ_matrix).size)
        # self.DFDZ_matrix = self.DFDZ_matrix.reshape(1,self.DFDZ_matrix.size)     #整个形
        #以上完成了DFDA到DFDZ的传播
        #接着完成DFDZ到DFDW的传播,进行推导后发现可以用矩阵乘法容易获得。
        temp = self.X_matrix.T
        self.DFDW_matrix = np.dot(temp , self.DFDZ_matrix)
        #完成DFDW的传播
        #接着完成DFDB的传播,进行推导后发现之间求DFDZ_matrix的和就好了。
        self.DFDB_matrix = np.sum(self.DFDZ_matrix,axis=0)

        #接着再完成DFDX的传播,进行推导后发现如下
        temp = self.W_matrix.T
        self.DFDX_matrix = np.dot(self.DFDZ_matrix , temp)
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def Dsigmoid(self,x):                     #sigmoid的逆函数
        out =  1 / (1 + np.exp(-x)) - (1 / (1 + np.exp(-x)))*(1 / (1 + np.exp(-x)))
        return out.reshape(x.shape)

    def Drelu(self,x):
        out = np.where(x >= 0, 1, 0)
        return out.reshape(x.shape)

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))
    #
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))


class lastlayer:
    # 定义基本属性
    Y_matrix = 0
    X_matrix = 0
    DFDY_matrix = 0
    DFDX_matrix = 0
    DYDX_matrix = 0
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __input_size = 0
    __output_size = 0
    __type = 0  # active fun type ; sigmod函数:0 ;  relu函数:1;
    def __init__(self,input_size,output_size,type=0):             #type表示了使用softmax函数来进行
        """
       example:  输入两个10元的向量，输出为两个1元的数据
       losslay = losslayer(input_size=(10,4),output_size=(10,4),type=0)
       """
        self.__input_size = input_size
        self.__output_size = output_size
        self.__type = type


    def set_X_matrix(self,input):
        """
               input.shape 必须等于__input_size,如果是一维数组也必须这样表示[[1,2,3]]
               """
        self.X_matrix = input
        if self.__input_size != self.X_matrix.shape:
            raise TypeError('bad operand type')
        pass

    def set_DFDY_matrix(self,input):
        """
                      input.shape 必须等于__output_size,如果是一维数组也必须这样表示[[1,2,3]]
                      """
        self.DFDY_matrix = input
        if self.__output_size != self.DFDY_matrix.shape:
            raise TypeError('bad operand type')
        pass

    def forward(self):
        if self.__type == 0:    #如果为softmax函数
            self.Y_matrix = self.softmax(self.X_matrix)


    def backward(self):
        if self.__type == 0:
            self.DYDX_matrix = self.Dsoftmax(self.X_matrix)
            temp = np.zeros(shape=(self.__input_size[0],self.__output_size[1]),dtype=float)      #初始化创建好要输出的矩阵信息
            for i in range(0,self.__input_size[0]):  #总共有。。行数据
                temp[i] = np.dot(self.DFDY_matrix[i] ,self.DYDX_matrix[i].T)
            self.DFDX_matrix = temp



    def softmax(self, a):
        c = np.max(a, axis=1)
        numtmp = c.size
        c = c.reshape(numtmp, 1)
        b = a - c
        exp_a = np.exp(b)
        sum_exp_a = np.sum(exp_a, axis=1)
        numtmp2 = sum_exp_a.size
        sum_exp_a = sum_exp_a.reshape(numtmp2, 1)
        y = exp_a / sum_exp_a
        return y


    def Dsoftmax(self, input):           #softmax函数的逆函数
        out_list = []       #创建一个空的list，用来存放偏导数矩阵列。   #由于一个输入样本就会有一个偏导数矩阵，所以多个样本就有队列了。
        for i in range(0,self.__input_size[0]):          #偏导数矩阵个数与样本数一样多
            a = input[i]
            a = a.reshape(1,a.size)
            out_DYDX_matrix = np.zeros(shape=(a.size,a.size),dtype=float)   #初始化一个偏导数矩阵
            # """
            #     偏导数矩阵类似如下：
            #
            #     dt1/dx1 dt2/dx1 dt3/dx1
            #     dt1/dx2 dt2/dx2 dt3/dx2
            #     dt1/dx3 dt2/dx3 dt3/dx3
            #
            #     """

            #求一些中间需要量
            c = np.max(a, axis=1)
            numtmp = c.size
            c = c.reshape(numtmp, 1)
            b = a - c
            exp_a = np.exp(b)
            sum_exp_a = np.sum(exp_a, axis=1)
            sum_exp_a_2 = sum_exp_a**2



            it = np.nditer(out_DYDX_matrix, flags=['multi_index'], op_flags=['readwrite'])    #通过这种方式来遍历numpy矩阵
            while not it.finished:
                index = it.multi_index
                if index[0] == index[1]:     #对于对角线元素
                    out_DYDX_matrix[index] = (exp_a[(0,index[1])]*(sum_exp_a - exp_a[(0,index[1])]))/sum_exp_a_2
                else:
                    out_DYDX_matrix[index] = (-exp_a[(0,index[1])] * exp_a[(0,index[0])])/sum_exp_a_2
                it.iternext()

            out_list.append(out_DYDX_matrix)

        return out_list

        # for index, element in enumerate(out_DYDX_matrix):           #根据求偏导结果
        #     # if index(0) == index(1):     #对于对角线元素
        #     #     element = (exp_a(index(1))*(sum_exp_a - exp_a(index(1))))/sum_exp_a_2
        #     # else:
        #     #     element = (-exp_a(index(1)) * exp_a(index(0)))/sum_exp_a_2


class losslayer:
    # 定义基本属性
    Y_matrix = 0      #前向计算的输出结果
    T_matrix = 0      #用于对比的正确结果
    DFDY_matrix = 0   #计算出来的反向传播的初始偏导数
    DFDT_matrix = 0   #计算出来的另一个偏导数矩阵（一般用不到）
    loss = 0          #输出的loss结果
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __input_size = 0
    __output_size = 0
    __type = 0  # loss fun type ; cross_entropy_error:0 ;  误差平方和函数:1;
    def __init__(self,input_size,output_size=(1,1),type=0):     #构造函数
        """
        example:  输入两个10元的向量，输出为两个1元的数据
        losslay = losslayer(input_size=(2,10),output_size=(1,1),type=0)
        """
        self.__input_size = input_size
        self.__output_size = output_size
        self.__type = type
        pass

    def set_T_matrix(self,T_matrix):
        self.T_matrix = T_matrix

    def set_Y_matrix(self,Y_matrix):
        self.Y_matrix = Y_matrix

    def forward(self):      #前向求最后的loss
        if self.__type == 0:        #如果为交叉项误差
            self.loss = self.cross_entropy_error(self.Y_matrix,self.T_matrix)
        elif self.__type == 1:      #如果为误差平方和
            self.loss = self.SSE(self.Y_matrix,self.T_matrix)


    def backward(self):     #反向传播求DFDY_matrix，计算出来的反向传播的初始偏导数
        if self.__type == 0:        #如果为交叉项误差
            self.DFDY_matrix , self.DFDT_matrix = self.Dcross_entropy_error(self.Y_matrix,self.T_matrix)
        elif self.__type == 1:      #如果为误差平方和
            self.DFDY_matrix , self.DFDT_matrix = self.DSSE(self.Y_matrix,self.T_matrix)



    def cross_entropy_error(self,y, t):          #交叉项误差
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def SSE(self,y,t):               #Sum of Squared Error 误差平方和
        out = np.sum((y-t)**2)
        return out

    def Dcross_entropy_error(self,y, t):       #交叉项函数的导函数，返回两组偏导数
        delta = 1e-7
        y = y + delta
        out_DFDY_matrix = -t * (1/y)          #通过展开列写可以验证
        out_DFDT_matrix = -(np.log(y))
        return (out_DFDY_matrix , out_DFDT_matrix)


    def DSSE(self,y,t):                        #误差平方和函数，返回两组偏导数
        out_DFDY_matrix = -2 * (t - y)
        out_DFDT_matrix = 2 * (t - y)
        return (out_DFDY_matrix , out_DFDT_matrix)






#test

# lay = layer(3,2,0)
# x = np.array([2,3,4])
# lay.set_X_matrix(x)
# DFDA_matrix = np.array([2,3])
# lay.set_DFDA_matrix(DFDA_matrix)
# lay.forward()
# lay.backward()
#
# print(lay.A_matrix)
# print(lay.DFDX_matrix)

# lastlay = lastlayer(4,4)
# x = np.array([2,3,4])
# DFDY_matrix = np.array([2,3,1])
# lastlay.set_X_matrix(x)
# lastlay.set_DFDY_matrix(DFDY_matrix)
# lastlay.forward()
# lastlay.backward()
# print(lastlay.Y_matrix)
# print(lastlay.DFDX_matrix)

# losslay = losslayer(input_size=5,output_size=5)
# y = np.array([0,0,0,0.2,0.8])
# t = np.array([0,0,0,0,1])
# losslay.set_T_matrix(t)
# losslay.set_Y_matrix(y)
# losslay.forward()
# losslay.backward()
# print(losslay.loss)
# print(losslay.DFDY_matrix)

#test for 多个输入
lay = layer(input_size=(2,3),output_size=(2,2),active_fun=0)
x = np.array([[2,3,4],[1,2,3]])
lay.set_X_matrix(x)
DFDA_matrix = np.array([[2,3],[1,1]])
lay.set_DFDA_matrix(DFDA_matrix)
lay.forward()
lay.backward()

print(lay.A_matrix)
print(lay.DFDX_matrix)
#
# lastlay = lastlayer(input_size=(3,4),output_size=(3,4),type=0)
# x = np.array([[2,3,4,5],[3,4,5,2],[2,2,1,2]])
# DFDY_matrix = np.array([[2,3,1,0],[2,2,1,2],[2,2,1,2]])
# # x = np.array([[2,3,4,5]])
# # DFDY_matrix = np.array([[2,3,1,4]])
# lastlay.set_X_matrix(x)
# lastlay.set_DFDY_matrix(DFDY_matrix)
# lastlay.forward()
# lastlay.backward()
# print(lastlay.Y_matrix)
# print(lastlay.DFDX_matrix)

# losslay = losslayer(input_size=(2,5),output_size=(2,1),type=0)
# y = np.array([[0,0,0,0.2,0.8],[0.1,0.9,0,0,0]])
# t = np.array([[0,0,0,0,1],[0,1,0,0,0]])
# losslay.set_T_matrix(t)
# losslay.set_Y_matrix(y)
# losslay.forward()
# losslay.backward()
# print(losslay.loss)
# print(losslay.DFDY_matrix)