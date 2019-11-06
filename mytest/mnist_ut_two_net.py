from mytest.two_net import *
from mongodb_data.mnist_data.mnist_data_config import *

if __name__ == '__main__':
    # 初始化网络
    mnist_ut_two_net = two_net(lay1_size=(784, 300), lay2_size=(300, 10), action_fun=1, loss_fun=0, batch_size=100,learn_rate=0.05)
    mnist_data = mnist_data()
    for x in range(2000):
        (imgs, tags) = mnist_data.get_train_data_from_mongodb(100)
        # 准备好图像数据
        input_data = np.zeros(shape=(100,784),dtype=float)
        for i in range(len(imgs)):
            input_data[i] =  imgs[i].reshape(1,imgs[i].size)/255.0

        #再准备好对标数据
        T_data = np.zeros(shape=(100, 10), dtype=float)
        for i in range(len(tags)):
            tmp = np.zeros(shape=(1,10),dtype=float)
            index = int(tags[i][0])
            tmp[0][index] = 1
            T_data[i] = tmp

        #开始执行前向以及后向跟新
        mnist_ut_two_net.set_input(input_data)
        mnist_ut_two_net.set_T_data(T_data)
        mnist_ut_two_net.forward()
        mnist_ut_two_net.predict()
        print(x)
        print(mnist_ut_two_net.value())

        mnist_ut_two_net.backward()
        mnist_ut_two_net.update_W_B()


    pass