# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_model(image, classes, shuffle=True, base_ch=144, groups=1, training=True):
    def channel_shuffle(net, output, group, scope="ChannelShuffle"):
        assert 0 == output % group, "Output channels must be a multiple of groups"
        num_channels_in_group = output // group
        with tf.variable_scope(scope):
            net = tf.split(net, output, axis=3, name="split")
            chs = []
            for i in range(group):
                for j in range(num_channels_in_group):
                    chs.append(net[i + j * group])
            net = tf.concat(chs, axis=3, name="concat")
        return net

    def group_conv(net, output, stride, group, relu=True, scope="GConv"):
        assert 0 == output % group, "Output channels must be a multiple of groups"
        num_channels_in_group = output // group
        with tf.variable_scope(scope):
            net = tf.split(net, group, axis=3, name="split")
            for i in range(group):
                net[i] = slim.conv2d(net[i],
                                     num_channels_in_group,
                                     [1, 1],
                                     stride=stride,
                                     activation_fn=tf.nn.relu if relu else None,
                                     normalizer_fn=slim.batch_norm,
                                     normalizer_params={'is_training': training})
            net = tf.concat(net, axis=3, name="concat")
        return net

    def shuffle_bottleneck(net, output, stride, group=1, scope="Unit"):
        if 1 != stride:
            _b, _h, _w, _c = net.get_shape().as_list()
            output = output - _c

        assert 0 == output % group, "Output channels must be a multiple of groups"

        with tf.variable_scope(scope):
            if 1 != stride:
                net_skip = slim.avg_pool2d(net, [3, 3], stride, padding="SAME", scope='3x3AVGPool')
            else:
                net_skip = net

            net = group_conv(net, output, 1, group, relu=True, scope="1x1ConvIn")

            if shuffle:
                net = channel_shuffle(net, output, group, scope="ChannelShuffle")

            with tf.variable_scope("3x3DWConv"):
                depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, output, 1],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.01))
                net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")
                # Todo: Add batch norm here

            net = group_conv(net, output, 1, group, scope="1x1ConvOut")

            if 1 != stride:
                net = tf.concat([net, net_skip], axis=3)
            else:
                net = net + net_skip

            net = tf.nn.relu(net)

        return net

    def shuffle_stage(net, output, repeat, group, scope="Stage"):
        with tf.variable_scope(scope):
            net = shuffle_bottleneck(net, output, 2, group, scope='Unit{}'.format(0))
            for i in range(repeat):
                net = shuffle_bottleneck(net, output, 1, group, scope='Unit{}'.format(i + 1))
        return net

    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with tf.variable_scope('Stage1'):
            net = slim.conv2d(image, 24, [3, 3], 2)
            net = slim.max_pool2d(net, [3, 3], 2, padding='SAME')

        net = shuffle_stage(net, base_ch, 3, groups, 'Stage2')
        net = shuffle_stage(net, base_ch * 2, 7, groups, 'Stage3')
        net = shuffle_stage(net, base_ch * 4, 3, groups, 'Stage4')

        with tf.variable_scope('Stage5'):
            net = tf.reduce_mean(net, [1, 2])
            net = slim.fully_connected(net, classes)

        return net