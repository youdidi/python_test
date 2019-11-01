import struct
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
abs_file=__file__
print("abs path is %s" %(__file__))
abs_dir=abs_file[:abs_file.rfind("\\")]     # windows下用\\分隔路径，linux下用/分隔路径
from mongoengine import *
connect('mnist',host='192.168.2.17')

class mnist_train_image(Document):
    tag = ListField(StringField(required=True))
    photo = FileField()

class mnist_t10k_image(Document):
    tag = ListField(StringField(required=True))
    photo = FileField()
class mnist_origin_data(Document):
    data_name = StringField(required=True)
    data = FileField()



class mnist_data:
    #读取训练用数据对应的标签
    def solve_train_lable_idx(self):
        f = open(abs_dir + '/train-labels-idx1-ubyte', "rb")
        # f = open(self.source_data_path + '/train-labels-idx1-ubyte', "rb")
        magic_number = struct.unpack(">I", f.read(4))[0]
        number_of_items = struct.unpack(">I", f.read(4))[0]
        lable_list = []
        count = 0
        while count < number_of_items:
            one_item = struct.unpack(">B", f.read(1))[0]
            lable_list.append(one_item)
            count = count+1
        return lable_list

    #读取验证用数据对应的标签
    def solve_t10k_lable_idx(self):
        f = open(abs_dir + '/t10k-labels-idx1-ubyte', "rb")
        # f = open(self.source_data_path + '/t10k-labels-idx1-ubyte', "rb")
        magic_number = struct.unpack(">I", f.read(4))[0]
        number_of_items = struct.unpack(">I", f.read(4))[0]
        lable_list = []
        count = 0
        while count < number_of_items:
            one_item = struct.unpack(">B", f.read(1))[0]
            lable_list.append(one_item)
            count = count+1
        return lable_list

    #读取训练用数据图像
    def solve_train_images_idx(self):
        f = open(abs_dir + '/train-images-idx3-ubyte', "rb")
        # f = open(self.source_data_path + '/train-images-idx3-ubyte', "rb")
        magic_number = struct.unpack(">I", f.read(4))[0]
        number_of_images = struct.unpack(">I", f.read(4))[0]
        number_of_rows = struct.unpack(">I", f.read(4))[0]
        number_of_columns = struct.unpack(">I", f.read(4))[0]
        images_list = np.fromfile(f, dtype=np.uint8)
        images_list = images_list.reshape(number_of_images, number_of_rows, number_of_columns)
        return images_list

    # 读取验证用数据图像
    def solve_t10k_images_idx(self):
        f = open(abs_dir + '/t10k-images-idx3-ubyte', "rb")
        # f = open(self.source_data_path + '/t10k-images-idx3-ubyte', "rb")
        magic_number = struct.unpack(">I", f.read(4))[0]
        number_of_images = struct.unpack(">I", f.read(4))[0]
        number_of_rows = struct.unpack(">I", f.read(4))[0]
        number_of_columns = struct.unpack(">I", f.read(4))[0]
        images_list = np.fromfile(f, dtype=np.uint8)
        images_list = images_list.reshape(number_of_images, number_of_rows, number_of_columns)
        return images_list

    # 上传数据到mongodb
    def upload_to_mongodb(self):
        # 首先上传训练数据
        train_lable_list = self.solve_train_lable_idx()
        train_images = self.solve_train_images_idx()
        count = 0
        while count < len(train_lable_list):
            oneobj = mnist_train_image()
            oneobj.tag = [str(train_lable_list[count]),'train']
            # misc.imsave('out.png',train_images[count])
            img = Image.fromarray(train_images[count])
            img.save('out.png')
            photo = open('out.png','rb')
            oneobj.photo.put(photo,content_type='png')
            oneobj.save()
            count = count + 1

        # 再上传验证数据
        t10k_lable_list = self.solve_t10k_lable_idx()
        t10k_images = self.solve_t10k_images_idx()
        count = 0
        while count < len(t10k_lable_list):
            oneobj = mnist_t10k_image()
            oneobj.tag = [str(t10k_lable_list[count]), 't10k']
            # misc.imsave('out.png',train_images[count])
            img = Image.fromarray(t10k_images[count])
            img.save('out.png')
            photo = open('out.png', 'rb')
            oneobj.photo.put(photo, content_type='png')
            oneobj.save()
            count = count + 1

        # 最后再上传原始数据
        #先上传t10k-images-idx3-ubyte
        oneobj = mnist_origin_data()
        oneobj.data_name = 't10k-images-idx3-ubyte'
        data = open(abs_dir + '/t10k-images-idx3-ubyte', "rb")
        oneobj.data.put(data, content_type='idx')
        oneobj.save()
        #再上传t10k-labels-idx1-ubyte
        oneobj = mnist_origin_data()
        oneobj.data_name = 't10k-labels-idx1-ubyte'
        data = open(abs_dir + '/t10k-labels-idx1-ubyte', "rb")
        oneobj.data.put(data, content_type='idx')
        oneobj.save()
        #再上传train-images-idx3-ubyte
        oneobj = mnist_origin_data()
        oneobj.data_name = 'train-images-idx3-ubyte'
        data = open(abs_dir + '/train-images-idx3-ubyte', "rb")
        oneobj.data.put(data, content_type='idx')
        oneobj.save()
        #再上传train-labels-idx1-ubyte
        oneobj = mnist_origin_data()
        oneobj.data_name = 'train-labels-idx1-ubyte'
        data = open(abs_dir + '/train-labels-idx1-ubyte', "rb")
        oneobj.data.put(data, content_type='idx')
        oneobj.save()

    # 从mongodb获取数据
    def get_train_data_from_mongodb(self,data_size):
        all_size = mnist_train_image.objects.count()
        all_list = range(0, all_size - 1, 1)
        rand_list = random.sample(all_list, data_size)
        output_img_list = []
        output_tag_list = []
        for oneobj_index in rand_list:
            oneobjs = mnist_train_image.objects[oneobj_index:oneobj_index + 1]
            for oneobj in oneobjs:
                tag = oneobj.tag
                data = oneobj.photo.read()  # 获取图片数据
                content_type = oneobj.photo.content_type
                outf = open('temp.' + content_type, 'wb')  # 创建文件
                outf.write(data)  # 存储图片
                outf.close()
                img = cv2.imread('temp.' + content_type, cv2.IMREAD_GRAYSCALE)
                output_img_list = output_img_list + [img]
                output_tag_list = output_tag_list + [tag]
        return (output_img_list, output_tag_list)

    def get_t10k_data_from_mongodb(self,data_size):
        all_size = mnist_t10k_image.objects.count()
        all_list = range(0, all_size - 1, 1)
        rand_list = random.sample(all_list, data_size)
        output_img_list = []
        output_tag_list = []
        for oneobj_index in rand_list:
            oneobjs = mnist_t10k_image.objects[oneobj_index:oneobj_index + 1]
            for oneobj in oneobjs:
                tag = oneobj.tag
                data = oneobj.photo.read()  # 获取图片数据
                content_type = oneobj.photo.content_type
                outf = open('temp.' + content_type, 'wb')  # 创建文件
                outf.write(data)  # 存储图片
                outf.close()
                img = cv2.imread('temp.' + content_type, cv2.IMREAD_GRAYSCALE)
                # img = cv2.imdecode(np.asarray(bytearray(data), dtype='uint8'), cv2.IMREAD_GRAYSCALE)
                output_img_list = output_img_list + [img]
                output_tag_list = output_tag_list + [tag]
        return (output_img_list, output_tag_list)



# #test

# single_mnist = mnist_data()
# single_mnist.upload_to_mongodb()



# oneobj.tag = [str(t10k_lable_list[count]), 't10k']
# # misc.imsave('out.png',train_images[count])
# img = Image.fromarray(t10k_images[count])
# img.save('out.png')
# photo = open('out.png', 'rb')
# photo = oneobj.photo.read()
# plt.imshow(photo)
# plt.show()
# type = oneobj.photo.content_type
# print(photo)
# print(type)
#单例
# mnist_data = mnist_data()
#
# (imgs, tags) = mnist_data.get_train_data_from_mongodb(100)
# print(len(imgs))
# print(len(tags))
# print(tags)

# (imgs, tags) = mnist_data.get_t10k_data_from_mongodb(10)
# print(len(imgs))
# print(len(tags))
# print(tags)


# all_size = mnist_train_image.objects.count()
# all_list = range(0, all_size-1, 1)
# rand_list = random.sample(all_list, 10)
# output_img_list = []
# output_tag_list = []
# for oneobj_index in rand_list:
#     oneobjs = mnist_train_image.objects[oneobj_index:oneobj_index+1]
#     for oneobj in oneobjs:
#         tag = oneobj.tag
#         data = oneobj.photo.read()  # 获取图片数据
#         content_type = oneobj.photo.content_type
#         outf = open('temp.'+ content_type, 'wb')  # 创建文件
#         outf.write(data)  # 存储图片
#         outf.close()
#         img = cv2.imread('temp.' + content_type, cv2.IMREAD_GRAYSCALE)
#         output_img_list = output_img_list + [img]
#         output_tag_list = output_tag_list + [tag]