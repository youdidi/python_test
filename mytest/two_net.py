from mytest.layer import *

class two_net:
    #普通变量
    input_data = 0
    T_data = 0
    output_predict = 0
    #私有变量
    __lay1_size = 0
    __lay2_size = 0
    __action_fun = 0
    __loss_fun = 0
    __batch_size = 0
    __learn_rate = 0
    def __init__(self,lay1_size,lay2_size,action_fun,loss_fun,batch_size,learn_rate):
        """
        构造函数
        :param lay1_size:   (780,20)
        :param lay2_size:   (20,10)
        :param action_fun: 0    代表类型
        :param loss_fun:  0     代表类型
        :param learn_rate:  0.1     代表学习率
        :param batch_size: 100
        """
        self.__lay1_size = lay1_size
        self.__lay2_size = lay2_size
        self.__action_fun = action_fun
        self.__loss_fun = loss_fun
        self.__batch_size = batch_size
        self.__learn_rate = learn_rate

        #初始化需要的每一层
        self.lay1 = layer(input_size=(self.__batch_size,self.__lay1_size[0]),output_size=(self.__batch_size,self.__lay1_size[1]),active_fun=self.__action_fun)
        self.lay2 = layer(input_size=(self.__batch_size,self.__lay2_size[0]),output_size=(self.__batch_size,self.__lay2_size[1]),active_fun=self.__action_fun)
        self.softmaxlay = lastlayer(input_size=(self.__batch_size,self.__lay2_size[1]),output_size=(self.__batch_size,self.__lay2_size[1]),type=0)
        self.losslay = losslayer(input_size=(self.__batch_size,self.__lay2_size[1]),output_size=(1,1),type=self.__loss_fun)
        pass

    def set_input(self,input):                 #设置原始输入数据
        self.input_data = input
        if self.input_data.shape != (self.__batch_size,self.__lay1_size[0]):
            raise TypeError('bad operand type')
        pass

    def set_T_data(self,t_data):                #设置输入数据的正确结果值
        self.T_data = t_data
        if self.T_data.shape != (self.__batch_size,self.__lay2_size[1]):
            raise TypeError('bad operand type')
        pass

    def forward(self):                 #进行一层一层的前向,一直到softmax输出,再到loss输出
        self.lay1.set_X_matrix(self.input_data)
        self.lay1.forward()
        self.lay2.set_X_matrix(self.lay1.A_matrix)
        self.lay2.forward()
        self.softmaxlay.set_X_matrix(self.lay2.A_matrix)
        self.softmaxlay.forward()
        self.losslay.set_Y_matrix(self.softmaxlay.Y_matrix)
        self.losslay.set_T_matrix(self.T_data)
        self.losslay.forward()
        pass

    def backward(self):
        self.losslay.backward()
        self.softmaxlay.set_DFDY_matrix(self.losslay.DFDY_matrix)
        self.softmaxlay.backward()
        self.lay2.set_DFDA_matrix(self.softmaxlay.DFDX_matrix)
        self.lay2.backward()
        self.lay1.set_DFDA_matrix(self.lay2.DFDX_matrix)
        self.lay1.backward()
        pass

    def update_W_B(self):                  #经过反向传播后，更新权重和偏置。注意：本函数必须在backward函数执行后才能执行。
        self.lay1.W_matrix = self.lay1.W_matrix - self.__learn_rate * self.lay1.DFDW_matrix
        self.lay2.W_matrix = self.lay2.W_matrix - self.__learn_rate * self.lay2.DFDW_matrix
        self.lay1.B_matrix = self.lay1.B_matrix - self.__learn_rate * self.lay1.DFDB_matrix
        self.lay2.B_matrix = self.lay2.B_matrix - self.__learn_rate * self.lay2.DFDB_matrix
        pass

    def predict(self):                     #经过前向传播后，再估计一个值。注意：本函数必须在forward函数执行后才能执行。
        out = np.zeros(self.softmaxlay.Y_matrix.shape)
        idx = self.softmaxlay.Y_matrix.argmax(axis=0)
        out[np.arange(self.softmaxlay.Y_matrix.shape[0]), idx[np.arange(self.softmaxlay.Y_matrix.shape[0])]] = 1
        self.output_predict = out
        pass

    def value(self):                        #计算本次batch的识别准确率
        count_right = 0
        for i in range(self.__batch_size):
            if self.output_predict[i] == self.T_data[i]   #如果预测结果正确，则正确结果+1
                count_right = count_right + 1

        return (count_right/self.__batch_size) * 100
        pass
