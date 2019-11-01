from mytest.two_net import *
from mongodb_data.mnist_data.mnist_data_config import *

if __name__ == '__main__':
    mnist_data = mnist_data()
    (imgs, tags) = mnist_data.get_train_data_from_mongodb(10)
    # 准备好图像数据
    input_data = np.zeros(shape=(10,784),dtype=float)
    for i in range(len(imgs)):
        input_data[i] =  imgs[i].reshape(1,imgs[i].size)

    #再准备好对标数据


    print(input_data[0])
    print(len(imgs))
    print(len(tags))
    pass