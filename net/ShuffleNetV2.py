import tensorflow as tf
import tensorflow.contrib as tc

slim = tc.slim

def shuffle_unit(x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x


def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x


def conv_bn(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x


def depthwise_conv_bn(x, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'depthwise_conv_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x


def resolve_shape(x):
    with tf.variable_scope(None, 'resolve_shape'):
        n, h, w, c = x.get_shape().as_list()
        if h is None or w is None:
            kernel_size = tf.convert_to_tensor([tf.shape(x)[1], tf.shape(x)[2]])
        else:
            kernel_size = [h, w]
    return kernel_size


def global_avg_pool2D(x):
    with tf.variable_scope(None, 'global_pool2D'):
        kernel_size = resolve_shape(x)
        x = slim.avg_pool2d(x, kernel_size, stride=1)
        x.set_shape([None, 1, 1, None])
    return x


def se_unit(x, bottleneck=2):
    with tf.variable_scope(None, 'SE_module'):
        n, h, w, c = x.get_shape().as_list()

        kernel_size = resolve_shape(x)
        x_pool = slim.avg_pool2d(x, kernel_size, stride=1)
        x_pool = tf.reshape(x_pool, shape=[-1, c])
        fc = slim.fully_connected(x_pool, bottleneck, activation_fn=tf.nn.relu,
                                  biases_initializer=None)
        fc = slim.fully_connected(fc, c, activation_fn=tf.nn.sigmoid,
                                  biases_initializer=None)
        if n is None:
            channel_w = tf.reshape(fc, shape=tf.convert_to_tensor([tf.shape(x)[0], 1, 1, c]))
        else:
            channel_w = tf.reshape(fc, shape=[n, 1, 1, c])

        x = tf.multiply(x, channel_w)
    return x


def shufflenet_v2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2):
    with tf.variable_scope(None, 'shuffle_v2_block'):
        if stride == 1:
            top, bottom = tf.split(x, num_or_size_splits=2, axis=3)

            half_channel = out_channel // 2

            top = conv_bn_relu(top, half_channel, 1)
            top = depthwise_conv_bn(top, kernel_size, stride, dilation)
            top = conv_bn_relu(top, half_channel, 1)

            out = tf.concat([top, bottom], axis=3)
            out = shuffle_unit(out, shuffle_group)

        else:
            half_channel = out_channel // 2
            b0 = conv_bn_relu(x, half_channel, 1)
            b0 = depthwise_conv_bn(b0, kernel_size, stride, dilation)
            b0 = conv_bn_relu(b0, half_channel, 1)

            b1 = depthwise_conv_bn(x, kernel_size, stride, dilation)
            b1 = conv_bn_relu(b1, half_channel, 1)

            out = tf.concat([b0, b1], axis=3)
            out = shuffle_unit(out, shuffle_group)
        return out

class ShuffleNetV2():

    first_conv_channel = 24

    def __init__(self, input_holder, cls, model_scale=1.0, shuffle_group=2, is_training=True):
        self.input = input_holder
        self.output = None
        self.cls = cls
        self.shuffle_group = shuffle_group
        self.channel_sizes = self._select_channel_size(model_scale)

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            self._build_model()

    def _select_channel_size(self, model_scale):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_scale == 0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_scale == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_scale == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_scale == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size.')

    def _build_model(self):
        with tf.variable_scope('init_block'):
            out = conv_bn_relu(self.input, self.first_conv_channel, 3, 2)
            out = slim.max_pool2d(out, 3, 2, padding='SAME')

        for idx, block in enumerate(self.channel_sizes[:-1]):
            with tf.variable_scope('shuffle_block_{}'.format(idx)):
                out_channel, repeat = block

                # First block is downsampling
                out = shufflenet_v2_block(out, out_channel, 3, 2, shuffle_group=self.shuffle_group)

                # Rest blocks
                for i in range(repeat-1):
                    out = shufflenet_v2_block(out, out_channel, 3, shuffle_group=self.shuffle_group)

        with tf.variable_scope('end_block'):
            out = conv_bn_relu(out, self.channel_sizes[-1][0], 1)

        with tf.variable_scope('prediction'):
            out = global_avg_pool2D(out)
            out = slim.conv2d(out, self.cls, 1, activation_fn=None, biases_initializer=None)
            out = tf.reshape(out, shape=[-1, self.cls])
            out = tf.identity(out, name='cls_prediction')
            self.output = out