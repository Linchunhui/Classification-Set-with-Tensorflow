# coding:utf-8
import os
import numpy as np
import tensorflow as tf
import glob
import sys
import argparse
from  read_batch import get_files, get_batch
import tensorflow.contrib.slim as slim
sys.path.append("project/task")

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

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--train_dir",
        default="D:/train",
        help="dataset dir")
    parser.add_argument(
        "--logs_train_dir",
        default='./model_save',
        help="log dir")
    parser.add_argument(
        "--N_CLASSES",
        default=3,
        help="The number of classes of dataset.")

    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--BATCH_SIZE",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=10,
        help="The number of train iterations.")
    parser.add_argument(
        "--init_lr",
        default=0.01,
        help="learning rate.")
    parser.add_argument(
        "--decay_steps",
        default=20,
        help="learning rate decay.")

    parser.add_argument(
        "--model",
        default="MoblieNetV3_small",
        help="model.")

    args = parser.parse_args()
    train(train_dir=args.train_dir,size=int(args.size),BATCH_SIZE=int(args.BATCH_SIZE),N_CLASSES=int(args.N_ClASSES),init_lr= \
          float(args.init_lr),decay_steps=int(args.decay_steps),logs_train_dir=args.logs_train_dir,epochs=int(args.epochs),model=args.model)


CAPACITY = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置最小gpu使用量

def train(train_dir,size,BATCH_SIZE,N_CLASSES,init_lr,decay_steps,logs_train_dir,epochs,model):
    train_list,train_label=get_files(train_dir)
    one_epoch_step = len(train_list) / BATCH_SIZE
    MAX_STEP=epochs * one_epoch_step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # label without one-hot
    batch_train, batch_labels = get_batch(train_list,train_label,size,size,BATCH_SIZE,CAPACITY)

    #net
    if model =="SqueezeNet":
        logits = squeeze_net(batch_train,classes=N_CLASSES)
    if model =="MobileNetV1":
        logits = MobileNetV1(batch_train, num_classes=N_CLASSES, is_training=True).output
    if model =="MobileNetV2":
        logits = MobileNetV2(batch_train, num_classes=N_CLASSES, is_training=True).output
    if model =="MobileNetV3_small":
        logits, _ = mobilenet_v3_small(batch_train, N_CLASSES, multiplier=1.0, is_training=True, reuse=None)
    if model == "MobileNetV3_large":
        logits, _ = mobilenet_v3_large(batch_train, N_CLASSES, multiplier=1.0, is_training = True, reuse = None)
    if model == "ShuffleNet":
        logits = get_model(batch_train, N_CLASSES)
    if model == "ShuffleNetV2":
        logits  = ShuffleNetV2(batch_train, N_CLASSES, model_scale=2.0, is_training=True).output
    if model == "Xception":
        logits = XceptionModel(batch_train,N_CLASSES, is_training=True)
    if model == "IGCV3":
        logits = IGCV3FPN(x=batch_train,num_classes=N_CLASSES, is_training=True).classifier_logits
    if model == "AlexNet":
        logits = alexnet(x=batch_train, keep_prob=0.5, num_classes=N_CLASSES)
    if model == "VGG-19":           #default vgg-19
        logits, _ = vgg_19(inputs=batch_train,num_classes=N_CLASSES, is_training=True, dropout_keep_prob=0.5)
    if model == "InceptionV1":
        logits, _ = inception_v1(inputs=batch_train, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=True)
    if model == "InceptionV2":
        logits, _ = inception_v2(inputs=batch_train, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=True)
    if model == "InceptionV3":
        logits, _ = inception_v3(inputs=batch_train, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=True)
    if model == "InceptionV4":
        logits, _ = inception_v4(inputs=batch_train, num_classes=N_CLASSES, dropout_keep_prob=0.5, is_training=True)
    if model == "ResNetV1":         #default resnet-101
        logits, _ = resnet_v1_101(inputs=batch_train, num_classes=N_CLASSES, is_training=True)
    if model == "ResNetV2":         #default resnet-101
        logits, _ = resnet_v2_101(inputs=batch_train, num_classes=N_CLASSES, is_training=True)
    if model == "Inception_ResNetV2":
        logits, _ = inception_resnet_v2(inputs=batch_train, num_classes=N_CLASSES, is_training=True)
    if model == "DenseNet":         #default densenet-121
        logits, _ = densenet121(inputs=batch_train, num_classes=N_CLASSES, is_training=True)
    if model == "SE_Inception_ResNetV2":
        logits = SE_Inception_resnet_v2(x=batch_train, classes_num=N_CLASSES, training=True)
    if model == "SE_InceptionV4":
        logits = SE_Inception_v4(x=batch_train, class_num=N_CLASSES, training=True)
    if model == "SE_ResNeXt":
        logits = SE_ResNeXt(x=batch_train, class_num=N_CLASSES, training=True)
    else:
        raise ValueError('Unsupported model .')

    print(logits.get_shape())

    # loss
    label_smoothing=0.1
    one_hot_labels = slim.one_hot_encoding(batch_labels, N_CLASSES)
    one_hot_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / N_CLASSES #标签平滑
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)   #sigmoid 多标签
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('train_loss', loss)
    # optimizer
    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=decay_steps,
                                    decay_rate=0.1)
    tf.summary.scalar('learning_rate', lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

    pred = tf.nn.sigmoid(logits)
    pred = tf.where(pred < 0.5, x=tf.zeros_like(pred), y=tf.ones_like(pred))

    result = tf.equal(pred, one_hot_labels)
    result2 = tf.reduce_mean(tf.cast(result, tf.float16), 1)
    correct = tf.equal(result2, tf.ones_like(result2))
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('train_acc', accuracy)

    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    # saver = tf.train.Saver()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

    #saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver.restore(sess, logs_train_dir+'/model.ckpt-10000')
    try:
        for step in range(int(MAX_STEP)):
            if coord.should_stop():
                break
            _, learning_rate, tra_loss, tra_acc = sess.run([optimizer, lr, loss, accuracy])
            if step % 10 == 0:
                print('Epoch %3d/%d, Step %6d/%d, lr %f, train loss = %.2f, train accuracy = %.2f%%' % (
                step / one_epoch_step, MAX_STEP / one_epoch_step, step, MAX_STEP, learning_rate, tra_loss,
                tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 5000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main(sys.argv)
