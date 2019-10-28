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
        self.__active_fun = active_fun
        self.__input_size = input_size
        self.__output_size = output_size
        self.W_matrix = np.random.randn(input_size,output_size)  #随机初始化W矩阵，权重矩阵
        self.B_matrix = np.random.randn(1,output_size)     #随机初始化B矩阵，偏置矩阵
        self.Z_matrix = np.zeros(shape=(1,output_size),dtype=float)     #初始化Z矩阵为全0矩阵
        self.A_matrix = np.zeros(shape=(1,output_size),dtype=float)     #初始化A矩阵为全0矩阵
        self.DFDZ_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDZ_matrix矩阵为全0矩阵
        self.DFDA_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDA_matrix矩阵为全0矩阵
        self.DFDW_matrix = np.zeros(shape=(input_size, output_size), dtype=float)  # 初始化DFDW_matrix矩阵为全0矩阵
        self.DFDB_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDB_matrix矩阵为全0矩阵
        self.DFDX_matrix = np.zeros(shape=(1, input_size), dtype=float)  # 初始化DFDX_matrix矩阵为全0矩阵
        self.DFDA_matrix = np.zeros(shape=(1, output_size), dtype=float)  # 初始化DFDA_matrix矩阵为全0矩阵

    def set_X_matrix(self,input_matrix):        #设置输入矩阵
        self.X_matrix = input_matrix


    def set_DFDA_matrix(self,DFDA_matrix):        #设置反向传输需要的偏导数矩阵
        self.DFDA_matrix = DFDA_matrix

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
        self.DFDZ_matrix = self.DFDZ_matrix.reshape(1,self.DFDZ_matrix.size)     #整个形
        #以上完成了DFDA到DFDZ的传播
        #接着完成DFDZ到DFDW的传播
        temp = self.X_matrix
        temp = temp.reshape(temp.size,1)
        temp = np.repeat(temp,self.Z_matrix.size)
        temp = temp.reshape(self.X_matrix.size,self.Z_matrix.size)
        DZDW_matrix = temp
        self.DFDW_matrix = temp * self.DFDZ_matrix
        #完成DFDW的传播
        #接着完成DFDB的传播
        DZDB_matrix = np.ones((1,self.B_matrix.size))
        self.DFDB_matrix = self.DFDZ_matrix * DZDB_matrix

        #接着再完成DFDX的传播
        DZDX_matrix = self.W_matrix.T
        self.DFDX_matrix = np.dot(self.DFDZ_matrix , DZDX_matrix)
        self.DFDX_matrix = self.DFDX_matrix.reshape(1,self.DFDX_matrix.size)
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
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __input_size = 0
    __output_size = 0
    __type = 0  # active fun type ; sigmod函数:0 ;  relu函数:1;
    def __init__(self，input_size,output_size,type=0):             #type表示了使用softmax函数来进行
        self.__input_size = input_size
        self.__output_size = output_size
        self.__type = type


    def set_X_matrix(self,input):
        self.X_matrix = input
        pass

    def set_DFDY_matrix(self,input):
        self.DFDY_matrix = input
        pass

    def forward(self):
        if type == 0:    #如果为softmax函数
            self.Y_matrix = self.softmax(self.X_matrix)


    def backward(self):

        pass


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


    def Dsoftmax(self, a):           #softmax函数的逆函数
        DYDX_matrix = np.zeros(shape=(self.__input_size,self.__output_size),dtype=float)   #初始化一个偏导数矩阵

        c = np.max(a, axis=1)
        numtmp = c.size
        c = c.reshape(numtmp, 1)
        b = a - c
        exp_a = np.exp(b)
        sum_exp_a = np.sum(exp_a, axis=1)
        sum_exp_a_2 = sum_exp_a**2


        for index, element in enumerate(DYDX_matrix):
            if index(0) == index(1):
                pass
            else：
                pass

            DYDX_matrix[index] =

        pass


#test

lay = layer(3,2,0)
x = np.array([2,3,4])
lay.set_X_matrix(x)
DFDA_matrix = np.array([2,3])
lay.set_DFDA_matrix(DFDA_matrix)
lay.forward()
lay.backward()

print(lay.A_matrix)
print(lay.DFDX_matrix)