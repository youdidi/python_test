from mytest.layer import *
from mytest.two_net import *
from mongodb_data.mnist_data.mnist_data_config import *


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad
def loss(mnist_ut_two_net):
    mnist_ut_two_net.forward()
    mnist_ut_two_net.predict()
    return mnist_ut_two_net.losslay.loss
    pass




if __name__ == '__main__':
    mnist_ut_two_net = two_net(lay1_size=(784, 20), lay2_size=(20, 10), action_fun=0, loss_fun=0, batch_size=1,
                               learn_rate=0.1)
    mnist_data = mnist_data()
    for x in range(2000):
        (imgs, tags) = mnist_data.get_train_data_from_mongodb(1)
        # 准备好图像数据
        input_data = np.zeros(shape=(1, 784), dtype=float)
        for i in range(len(imgs)):
            input_data[i] = imgs[i].reshape(1, imgs[i].size)

        # 再准备好对标数据
        T_data = np.zeros(shape=(1, 10), dtype=float)
        for i in range(len(tags)):
            tmp = np.zeros(shape=(1, 10), dtype=float)
            index = int(tags[i][0])
            tmp[0][index] = 1
            T_data[i] = tmp

        # 开始执行前向以及后向跟新
        mnist_ut_two_net.set_input(input_data)
        mnist_ut_two_net.set_T_data(T_data)
        mnist_ut_two_net.forward()
        mnist_ut_two_net.predict()
        # print(x)
        # print(mnist_ut_two_net.value())

        mnist_ut_two_net.backward()
        # mnist_ut_two_net.update_W_B()


        # print(mnist_ut_two_net.lay1.X_matrix)
        # print(mnist_ut_two_net.lay1.W_matrix)
        # print(mnist_ut_two_net.lay1.B_matrix)
        # print(mnist_ut_two_net.lay1.DFDA_matrix)
        # print(mnist_ut_two_net.lay1.DFDZ_matrix)
        print('lay1.DFDW_matrix',mnist_ut_two_net.lay1.DFDW_matrix)
        # print(mnist_ut_two_net.lay1.DFDX_matrix)
        print('lay1.DFDB_matrix',mnist_ut_two_net.lay1.DFDB_matrix)
        # print(mnist_ut_two_net.lay1.Z_matrix)
        # print(mnist_ut_two_net.lay1.A_matrix)
        #
        # print(mnist_ut_two_net.lay2.X_matrix)
        # print(mnist_ut_two_net.lay2.W_matrix)
        # print(mnist_ut_two_net.lay2.B_matrix)
        # print(mnist_ut_two_net.lay2.DFDA_matrix)
        # print(mnist_ut_two_net.lay2.DFDZ_matrix)
        # print(mnist_ut_two_net.lay2.DFDW_matrix)
        # print(mnist_ut_two_net.lay2.DFDX_matrix)
        # print(mnist_ut_two_net.lay2.DFDB_matrix)
        # print(mnist_ut_two_net.lay2.Z_matrix)
        # print(mnist_ut_two_net.lay2.A_matrix)
        #
        # print('softmaxlay.X_matrix',mnist_ut_two_net.softmaxlay.X_matrix)
        # print('softmaxlay.Y_matrix',mnist_ut_two_net.softmaxlay.Y_matrix)
        # print('softmaxlay.DFDX_matrix',mnist_ut_two_net.softmaxlay.DFDX_matrix)
        # print('softmaxlay.DFDY_matrix',mnist_ut_two_net.softmaxlay.DFDY_matrix)
        # print('softmaxlay.DYDX_matrix',mnist_ut_two_net.softmaxlay.DYDX_matrix)

        # print('losslayer.Y_matrix', mnist_ut_two_net.losslay.Y_matrix)
        # print('losslayer.T_matrix',mnist_ut_two_net.losslay.T_matrix)
        # print('losslayer.DFDY_matrix',mnist_ut_two_net.losslay.DFDY_matrix)
        # print('losslayer.DFDT_matrix',mnist_ut_two_net.losslay.DFDT_matrix)
        # print('loss',mnist_ut_two_net.losslay.loss)


        #利用数值法求梯度，进行比较
        f = lambda w: loss(mnist_ut_two_net)
        dW = numerical_gradient(f, mnist_ut_two_net.lay1.W_matrix)
        print(dW)

        pass

