from mytest.two_net import *
from mongodb_data.mnist_data.mnist_data_config import *

if __name__ == '__main__':
    mnist_data = mnist_data()
    (imgs, tags) = mnist_data.get_train_data_from_mongodb(100)
    print(len(imgs))
    print(len(tags))
    pass