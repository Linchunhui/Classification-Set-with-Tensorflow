# coding:utf-8
import os
import numpy as np
import tensorflow as tf
import glob
import sys
from nets.mobilenet import mobilenet_v2
import tensorflow.contrib.slim as slim
#sys.path.append("project/slim")

def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
            image_list.append(img)
            label_list.append(int(label_dict[label]))
    print('There are %d data' % (len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list


label_dict, label_dict_res = {}, {}
# 手动指定一个从类别到label的映射关系
with open("label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    #sess=tf.Session()
    #print(sess.run(image_contents))
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 数据增强
    # image = tf.image.resize_image_with_pad(image, target_height=image_W, target_width=image_H)
    image = tf.image.resize_images(image, (image_W, image_H))
    # 随机左右翻转
    image = tf.image.random_flip_left_right(image)
    # 随机上下翻转
    image = tf.image.random_flip_up_down(image)
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=32 / 255.0)
    # 随机设置图片的对比度
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 随机设置图片的色度
    image = tf.image.random_hue(image, max_delta=0.05)
    # 随机设置图片的饱和度
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # 标准化,使图片的均值为0，方差为1
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    tf.summary.image("input_img", image_batch, max_outputs=5)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

