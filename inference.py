import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
import argparse
import sys
#net
from net.MobileNetV1 import MobileNetV1
from net.MobileNetV2 import MobileNetV2
from net.MobileNetV3 import mobilenet_v3_small, mobilenet_v3_large
from net.ShuffleNet import get_model
from net.ShuffleNetV2 import ShuffleNetV2
from net.SqueezeNet import squeeze_net
from net.Xception import XceptionModel
from net.IGCV3 import IGCV3FPN

from net.AlexNet import alexnet
from net.Vgg import vgg_a,vgg_16,vgg_19
from net.InceptionV1 import inception_v1
from net.InceptionV2 import inception_v2
from net.InceptionV3 import inception_v3
from net.InceptionV4 import inception_v4
from net.Inception_ResNetV2 import inception_resnet_v2
from net.ResNetV1 import resnet_v1_50, resnet_v1_101, resnet_v1_152, resnet_v1_200
from net.ResNetV2 import resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200
from net.DenseNet import densenet121, densenet161, densenet169
from net.SE_Inception_ResNetV2 import SE_Inception_resnet_v2
from net.SE_InceptionV4 import SE_Inception_v4
from net.SE_ResNeXt import SE_ResNeXt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置最小gpu使用量

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--data_dir",
        default="D:/RoadMapSample/",
        help="data dir")
    parser.add_argument(
        "--save_dir",
        default="D:/res/'",
        help="save dir")
    parser.add_argument(
        "--logs_train_dir",
        default='./model_save/model.ckpt-24000',
        help="log dir")
    parser.add_argument(
        "--N_CLASSES",
        default=3,
        help="The number of classes of dataset.")
    parser.add_argument(
        "--cate",
        default=['cat','dog','other'],
        help="cate.")
    parser.add_argument(
        "--model",
        default="MoblieNetV3_small",
        help="model.")

    args = parser.parse_args()
    init_tf(args.logs_train_dir, args.model, int(args.size), int(args.N_CLASSES))
    read_file_all(args.data_dir, cate=args.cate, size=int(args.size), dir=args.save_dir)

def init_tf(logs_train_dir,model,size,N_CLASSES):
    global sess, pred, x
    # process image

    x = tf.placeholder(tf.float32, shape=[size, size, 3])
    x_norm = tf.image.per_image_standardization(x)
    x_4d = tf.reshape(x_norm, [-1, size, size, 3])
    # predict
    # net
    if model == "SqueezeNet":
        logits = squeeze_net(x_4d, classes=N_CLASSES)
    if model == "MobileNetV1":
        logits = MobileNetV1(x_4d, num_classes=N_CLASSES, is_training=False).output
    if model == "MobileNetV2":
        logits = MobileNetV2(x_4d, num_classes=N_CLASSES, is_training=False).output
    if model == "MobileNetV3_small":
        logits, _ = mobilenet_v3_small(x_4d, N_CLASSES, multiplier=1.0, is_training=False, reuse=None)
    if model == "MobileNetV3_large":
        logits, _ = mobilenet_v3_large(x_4d, N_CLASSES, multiplier=1.0, is_training=False, reuse=None)
    if model == "ShuffleNet":
        logits = get_model(x_4d, N_CLASSES)
    if model == "ShuffleNetV2":
        logits = ShuffleNetV2(x_4d, N_CLASSES, model_scale=2.0, is_training=False).output
    if model == "Xception":
        logits = XceptionModel(x_4d, N_CLASSES, is_training=False)
    if model == "IGCV3":
        logits = IGCV3FPN(x=x_4d, num_classes=N_CLASSES, is_training=False).classifier_logits
    if model == "AlexNet":
        logits = alexnet(x=x_4d, keep_prob=0.5, num_classes=N_CLASSES)
    if model == "VGG-19":           #default vgg-19
        logits, _ = vgg_19(inputs=x_4d,num_classes=N_CLASSES, is_training=False, dropout_keep_prob=0.5)
    if model == "InceptionV1":
        logits, _ = inception_v1(inputs=x_4d, num_classes=N_CLASSES,dropout_keep_prob=0.5, is_training=False)
    if model == "InceptionV2":
        logits, _ = inception_v2(inputs=x_4d, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=False)
    if model == "InceptionV3":
        logits, _ = inception_v3(inputs=x_4d, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=False)
    if model == "InceptionV4":
        logits, _ = inception_v4(inputs=x_4d, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=False)
    if model == "ResNetV1":         #default resnet-101
        logits, _ = resnet_v1_101(inputs=x_4d, num_classes=N_CLASSES, is_training=False)
    if model == "ResNetV2":         #default resnet-101
        logits, _ = resnet_v2_101(inputs=x_4d, num_classes=N_CLASSES, is_training=False)
    if model == "Inception_ResNetV2":
        logits, _ = inception_resnet_v2(inputs=x_4d, num_classes=N_CLASSES, is_training=False)
    if model == "DenseNet":         #default densenet-121
        logits, _ = densenet121(inputs=x_4d, num_classes=N_CLASSES, is_training=False)
    if model == "SE_Inception_ResNetV2":
        logits = SE_Inception_resnet_v2(x=x_4d,classes_num=N_CLASSES,training=False)
    if model == "SE_InceptionV4":
        logits = SE_Inception_v4(x=x_4d, class_num=N_CLASSES, training=False)
    if model == "SE_ResNeXt":
        logits = SE_ResNeXt(x=x_4d, class_num=N_CLASSES, training=False)
    else:
        raise ValueError('Unsupported model .')

    print("logit", np.shape(logits))
    pred = tf.nn.sigmoid(logits)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, logs_train_dir)
    print('load model done...')


def evaluate_image2(img_path,size):
    img = image.load_img(img_path, target_size=(size, size))
    image_array = image.img_to_array(img)
    prediction = sess.run(pred, feed_dict={x: image_array})
    return prediction

def func(a):
    for i in range(len(a)):
        # a[i]=1 if a[i]==True else 0
        a[i] = int(a[i])
    return a

def output1(pred, threshold=0.4):
    pred = pred[0]
    flag = [i >= threshold for i in pred]
    a = func(flag)
    return a

def output(pred,cate,threshold=0.4):
    pred = pred[0]
    flag1 = [i >= threshold for i in pred]
    res1 = []
    for i in range(len(flag1)):
        if flag1[i] == True:
            res1.append(cate[i])
    return res1

def read_file_all(data_dir_path,cate,size,dir):
    file_list = []
    res_list = []
    res_cate_list = []
    result = pd.DataFrame()
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    i = 0
    for f in os.listdir(data_dir_path):
        if f == 'Thumbs.db':
            pass
        image_path = os.path.join(data_dir_path, f)
        # print(i,image_path)
        i += 1
        if os.path.isfile(image_path):
            preds = evaluate_image2(image_path,size)
            res_int = output1(preds)  # 数字
            res_cate = output(preds,cate)  # 类别
            print(i, image_path, preds, res_cate, res_int)
            # print(res_int)
            file_list.append(image_path)
            res_list.append(res_int)
            res_cate_list.append(res_cate)
            # count=0
            '''if res_cate == ['cat', 'dog']:
                count += 1
            elif res_cate == ['cat']:
                count1 += 1
            elif res_cate == ['dog']:
                count2 += 1
            else:
                count3 += 1'''
    result['filename'] = file_list
    result['res_int'] = res_list
    result['res_cate'] = res_cate_list
    name='result.csv'
    result.to_csv(os.path.join(dir,name), index=False, encoding='utf-8')
    #print('cat cout:{} dog count:{},cat_dog count:{} other:{}/all count:{}'.format(count1, count2, count, count3,
                                                                                #   len(res_list)))


if __name__ == "__main__":
    main(sys.argv)