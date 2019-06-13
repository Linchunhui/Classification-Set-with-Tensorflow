import tensorflow as tf
from tensorflow.python.ops import array_ops
from read_batch import get_batch
Detection_or_Classifier = 'classifier'  # 'detection','classifier'


class IGCV3FPN():
    MEAN = [103.94, 116.78, 123.68]
    NORMALIZER = 0.017

    def __init__(self,x, num_classes,
                 learning_rate=0.001,is_training=True):
        self.input = x
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.__build()

    def __build(self):
        self.norm = 'batch_norm'  # group_norm,batch_norm
        self.activate = 'relu6'  # selu,leaky,swish,relu,relu6
        self.BlockInfo = {  # none,none/2,none/2,32
            '1': [1, 16, 1, 1, 2, True],  # ratio,num_filters,repeat,stride,group,use_depthwise
            # none,none/2,none/2,16
            '2': [6, 24, 4, 2, 2, True],
            # none,none/4,none/4,24
            '3': [6, 32, 6, 2, 2, True],
            # none,none/8,none/8,32
            '4': [6, 64, 8, 2, 2, True],
            # none,none/16,none/16,64
            '5': [6, 96, 6, 1, 2, True],
            # none,none/16,none/16,96
            '6': [6, 160, 6, 2, 2, True],
            # none,none/32,none/32,160
            '7': [6, 320, 1, 1, 2, True],
            # none,none/32,none/32,320
            '8': [1, 1280, 1, 1, 2, False]
            # none,none/32,none/32,1280
        }

        with tf.variable_scope('zsc_preprocessing'):
            red, green, blue = tf.split(self.input, num_or_size_splits=3, axis=3)
            x = tf.concat([
                tf.subtract(blue, IGCV3FPN.MEAN[0]) * IGCV3FPN.NORMALIZER,
                tf.subtract(green, IGCV3FPN.MEAN[1]) * IGCV3FPN.NORMALIZER,
                tf.subtract(red, IGCV3FPN.MEAN[2]) * IGCV3FPN.NORMALIZER,
            ], 3)

        with tf.variable_scope('zsc_feature'):
            # none,none,none,3
            x = PrimaryConv('PrimaryConv', x, 32, self.norm, self.activate, self.is_training)
            skip_0 = x
            # none,none/2,none/2,32

            x = IGCV3Block('IGCV3_1', x, self.BlockInfo['1'][0], self.BlockInfo['1'][1], self.BlockInfo['1'][2],
                           self.BlockInfo['1'][3], self.BlockInfo['1'][4],
                           self.BlockInfo['1'][5], self.norm, self.activate, self.is_training)
            skip_1 = x
            # none,none/2,none/2,16

            x = IGCV3Block('IGCV3_2', x, self.BlockInfo['2'][0], self.BlockInfo['2'][1], self.BlockInfo['2'][2],
                           self.BlockInfo['2'][3], self.BlockInfo['2'][4],
                           self.BlockInfo['2'][5], self.norm, self.activate, self.is_training)
            skip_2 = x
            # none,none/4,none/4,24

            x = IGCV3Block('IGCV3_3', x, self.BlockInfo['3'][0], self.BlockInfo['3'][1], self.BlockInfo['3'][2],
                           self.BlockInfo['3'][3], self.BlockInfo['3'][4],
                           self.BlockInfo['3'][5], self.norm, self.activate, self.is_training)
            skip_3 = x
            # none,none/8,none/8,32

            x = IGCV3Block('IGCV3_4', x, self.BlockInfo['4'][0], self.BlockInfo['4'][1], self.BlockInfo['4'][2],
                           self.BlockInfo['4'][3], self.BlockInfo['4'][4],
                           self.BlockInfo['4'][5], self.norm, self.activate, self.is_training)
            skip_4 = x
            # none,none/16,none/16,64

            x = IGCV3Block('IGCV3_5', x, self.BlockInfo['5'][0], self.BlockInfo['5'][1], self.BlockInfo['5'][2],
                           self.BlockInfo['5'][3], self.BlockInfo['5'][4],
                           self.BlockInfo['5'][5], self.norm, self.activate, self.is_training)
            skip_5 = x
            # none,none/16,none/16,96

            x = IGCV3Block('IGCV3_6', x, self.BlockInfo['6'][0], self.BlockInfo['6'][1], self.BlockInfo['6'][2],
                           self.BlockInfo['6'][3], self.BlockInfo['6'][4],
                           self.BlockInfo['6'][5], self.norm, self.activate, self.is_training)
            skip_6 = x
            # none,none/32,none/32,160

            x = IGCV3Block('IGCV3_7', x, self.BlockInfo['7'][0], self.BlockInfo['7'][1], self.BlockInfo['7'][2],
                           self.BlockInfo['7'][3], self.BlockInfo['7'][4],
                           self.BlockInfo['7'][5], self.norm, self.activate, self.is_training)
            skip_7 = x
            # none,none/32,none/32,320

            x = IGCV3Block('IGCV3_8', x, self.BlockInfo['8'][0], self.BlockInfo['8'][1], self.BlockInfo['8'][2],
                           self.BlockInfo['8'][3], self.BlockInfo['8'][4],
                           self.BlockInfo['8'][5], self.norm, self.activate, self.is_training)
            skip_8 = x
            # none,none/32,none/32,1280

        if Detection_or_Classifier == 'classifier':
            with tf.variable_scope('zsc_classifier'):
                global_pool = tf.reduce_mean(x, [1, 2], keep_dims=True)
                self.classifier_logits = tf.reshape(
                    _conv_block('Logits', global_pool, self.num_classes, 1, 1, 'SAME', self.norm, self.activate,
                                self.is_training),
                    [tf.shape(global_pool)[0], self.num_classes])

################################################################################################################
################################################################################################################
################################################################################################################
##IGCV3Block
def IGCV3Block(name, x, ratio=6, num_filters=32, repeat=1, stride=1, group=2, use_depthwise=True, norm='group_norm',
               activate='selu', is_training=True):
    with tf.variable_scope(name):
        for i in range(repeat):
            input = x

            x = _group_conv('group_conv_{}_0'.format(i), x, group, ratio * num_filters, 1, 1, 'SAME', norm, activate,
                            is_training)
            if use_depthwise:
                if stride == 2:
                    if i == 0:
                        x = _depthwise_conv2d('depthwise_{}_0'.format(i), x, 1, 3, 2, 'SAME', norm, activate,
                                              is_training)
                    else:
                        x = _depthwise_conv2d('depthwise_{}_0'.format(i), x, 1, 3, 1, 'SAME', norm, activate,
                                              is_training)
                else:
                    x = _depthwise_conv2d('depthwise_{}_0'.format(i), x, 1, 3, 1, 'SAME', norm, activate, is_training)
            x = _group_conv('group_conv_{}_1'.format(i), x, group, num_filters, 1, 1, 'SAME', norm, None, is_training)

            if stride == 1:
                if input.get_shape().as_list()[-1] == x.get_shape().as_list()[-1]:
                    pass
                else:
                    input = _conv_block('conv_{}_2'.format(i), input, num_filters, 1, 1, 'SAME', norm, activate,
                                        is_training)
                x += input
            else:
                pass
        return x


##selfattention
def SelfAttention(name, x, norm='group_norm', activate='selu', is_training=True):
    with tf.variable_scope(name):
        C = x.get_shape().as_list()[-1]
        f = _conv_block('f', x, C // 8, 1, 1, 'SAME', norm, activate, is_training)
        g = _conv_block('g', x, C // 8, 1, 1, 'SAME', norm, activate, is_training)
        h = _conv_block('h', x, C, 1, 1, 'SAME', norm, activate, is_training)

        f = tf.transpose(f, [0, 3, 2, 1])
        g = tf.transpose(g, [0, 3, 1, 2])
        h = tf.transpose(h, [0, 3, 1, 2])

        attention = tf.multiply(f, g)
        attention = tf.reduce_mean(attention, [1], keep_dims=True)
        attention = tf.nn.softmax(attention)

        weight = tf.Variable(0.0, trainable=True)
        attention = weight * tf.multiply(h, attention)
        attention = tf.transpose(attention, [0, 2, 3, 1])

        return x + attention


##primary_conv
def PrimaryConv(name, x, num_filters=32, norm='group_norm', activate='selu', is_training=True):
    with tf.variable_scope(name):
        # none,none,none,3
        x = _conv_block('conv', x, num_filters, 3, 2, 'SAME', norm, activate,
                        is_training)  # none,none/2,none/2,num_filters

        return x


##_conv_block
def _conv_block(name, x, num_filters=16, kernel_size=3, stride=2, padding='SAME', norm='group_norm', activate='selu',
                is_training=True):
    with tf.variable_scope(name):
        w = GetWeight('weight', [kernel_size, kernel_size, x.shape.as_list()[-1], num_filters])
        x = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding, name='conv')

        if norm == 'batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001, name='batchnorm')
        elif norm == 'group_norm':
            x = group_norm(x, name='groupnorm')
        else:
            b = tf.get_variable('bias', num_filters, tf.float32, initializer=tf.constant_initializer(0.001))
            x += b
        if activate == 'leaky':
            x = LeakyRelu(x, leak=0.1, name='leaky')
        elif activate == 'selu':
            x = selu(x, name='selu')
        elif activate == 'swish':
            x = swish(x, name='swish')
        elif activate == 'relu':
            x = tf.nn.relu(x, name='relu')
        elif activate == 'relu6':
            x = tf.nn.relu6(x, name='relu6')
        else:
            pass

        return x


##_depthwise_conv2d
def _depthwise_conv2d(name, x, scale=1, kernel_size=3, stride=1, padding='SAME', norm='group_norm', activate='selu',
                      is_training=True):
    with tf.variable_scope(name) as scope:
        w = GetWeight('weight', [kernel_size, kernel_size, x.shape.as_list()[-1], scale])
        x = tf.nn.depthwise_conv2d(x, w, [1, stride, stride, 1], padding)

        if norm == 'batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001, name='batchnorm')
        elif norm == 'group_norm':
            x = group_norm(x, name='groupnorm')
        else:
            b = tf.get_variable('bias', scale, tf.float32, initializer=tf.constant_initializer(0.001))
            x += b
        if activate == 'leaky':
            x = LeakyRelu(x, leak=0.1, name='leaky')
        elif activate == 'selu':
            x = selu(x, name='selu')
        elif activate == 'swish':
            x = swish(x, name='swish')
        elif activate == 'relu':
            x = tf.nn.relu(x, name='relu')
        elif activate == 'relu6':
            x = tf.nn.relu6(x, name='relu6')
        else:
            pass
        return x


##_group_conv with channel shuffle use depthwise_conv2d
def _group_conv(name, x, group=4, num_filters=16, kernel_size=1, stride=1, padding='SAME', norm='group_norm',
                activate='selu', is_training=True):
    with tf.variable_scope(name):
        C = x.shape.as_list()[-1]
        num_012 = tf.shape(x)[:3]
        assert C % group == 0 and num_filters % group == 0

        w = GetWeight('weight', [kernel_size, kernel_size, C, num_filters // group])
        x = tf.nn.depthwise_conv2d(x, w, [1, stride, stride, 1], padding)

        x = tf.reshape(x, tf.concat([[num_012[0]], tf.cast(num_012[1:3] / kernel_size, tf.int32),
                                     tf.cast([group, C // group, num_filters // group], tf.int32)], axis=-1))
        x = tf.reduce_sum(x, axis=4)
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, tf.concat(
            [[num_012[0]], tf.cast(num_012[1:3] / kernel_size, tf.int32), tf.cast([num_filters], tf.int32)], axis=-1))

        if norm == 'batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001, name='batchnorm')
        elif norm == 'group_norm':
            x = group_norm(x, name='groupnorm')
        else:
            pass
        if activate == 'leaky':
            x = LeakyRelu(x, leak=0.1, name='leaky')
        elif activate == 'selu':
            x = selu(x, name='selu')
        elif activate == 'swish':
            x = swish(x, name='swish')
        elif activate == 'relu':
            x = tf.nn.relu(x, name='relu')
        elif activate == 'relu6':
            x = tf.nn.relu6(x, name='relu6')
        else:
            pass

        return x


##SEAttention
def SEAttention(name, x_high, x_low, downsample=True, norm='group_norm', activate='selu', is_training=True):
    with tf.variable_scope(name):
        x_high_C = x_high.get_shape().as_list()[-1]
        x_low_C = x_low.get_shape().as_list()[-1]

        if downsample:
            # x_low = tf.nn.avg_pool(x_low,[1,2,2,1],[1,2,2,1],'SAME')
            x_low = _conv_block('pool', x_low, x_low_C, 3, 2, 'SAME', norm, activate, is_training)
        else:
            pass

        x = tf.concat([x_high, x_low], axis=-1)
        x = SE('SE', x) * x
        x = _conv_block('conv', x, x_high_C, 1, 1, 'SAME', norm, None)
        weight = tf.Variable(0.0, trainable=True)

        return x_high + x * weight


##senet
def SE(name, x):
    with tf.variable_scope(name):
        # none,none,none,C
        C = x.get_shape().as_list()[-1]
        # SEnet channel attention
        weight_c = tf.reduce_mean(x, [1, 2], keep_dims=True)  # none,1,1,C
        weight_c = _conv_block('conv_1', weight_c, C // 8, 1, 1, 'SAME', None, 'relu')
        weight_c = _conv_block('conv_2', weight_c, C, 1, 1, 'SAME', None, None)

        weight_c = tf.nn.sigmoid(weight_c)  # none,1,1,C

        return weight_c


##weight variable
def GetWeight(name, shape, weights_decay=0.0001):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', shape, tf.float32, initializer=VarianceScaling())
        weight_decay = tf.multiply(tf.nn.l2_loss(w), weights_decay, name='weight_loss')
        tf.add_to_collection('regularzation_loss', weight_decay)
        return w


##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math


def _compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
        if scale <= 0.:
            raise ValueError("`scale` must be positive float.")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)
        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        scale = self.scale
        scale_shape = shape
        if partition_info is not None:
            scale_shape = partition_info.full_shape
        fan_in, fan_out = _compute_fans(scale_shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if self.distribution == "normal":
            stddev = math.sqrt(scale)
            return random_ops.truncated_normal(shape, 0.0, stddev,
                                               dtype, seed=self.seed)
        else:
            limit = math.sqrt(3.0 * scale)
            return random_ops.random_uniform(shape, -limit, limit,
                                             dtype, seed=self.seed)


##group_norm
def _max_divisible(input, max=1):
    for i in range(1, max + 1)[::-1]:
        if input % i == 0:
            return i


def group_norm(x, eps=1e-5, name='group_norm'):
    with tf.variable_scope(name):
        _, _, _, C = x.get_shape().as_list()
        G = _max_divisible(C, max=C // 2 + 1)
        G = min(G, C)
        if C % 32 == 0:
            G = min(G, 32)

        # group_list = tf.split(tf.expand_dims(x,axis=3),num_or_size_splits=G,axis=4)#[(none,none,none,1,C//G),...]
        # x = tf.concat(group_list,axis=3)#none,none,none,G,C//G
        x = tf.reshape(x, tf.concat([tf.shape(x)[:3], tf.constant([G, C // G])], axis=0))  # none,none,none,G,C//G

        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)  # none,none,none,G,C//G
        x = (x - mean) / tf.sqrt(var + eps)  # none,none,none,G,C//G

        # group_list = tf.split(x,num_or_size_splits=G,axis=3)#[(none,none,none,1,C//G),...]
        # x = tf.squeeze(tf.concat(group_list,axis=4),axis=3)#none,none,none,C
        x = tf.reshape(x, tf.concat([tf.shape(x)[:3], tf.constant([C])], axis=0))  # none,none,none,C

        gamma = tf.Variable(tf.ones([C]), name='gamma')
        beta = tf.Variable(tf.zeros([C]), name='beta')
        gamma = tf.reshape(gamma, [1, 1, 1, C])
        beta = tf.reshape(beta, [1, 1, 1, C])

    return x * gamma + beta


##LeakyRelu
def LeakyRelu(x, leak=0.1, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


##selu
def selu(x, name='selu'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


##swish
def swish(x, name='swish'):
    with tf.variable_scope(name):
        beta = tf.Variable(1.0, trainable=True)
        return x * tf.nn.sigmoid(beta * x)


##crelu 注意使用时深度要减半
def crelu(x, name='crelu'):
    with tf.variable_scope(name):
        x = tf.concat([x, -x], axis=-1)
        return tf.nn.relu(x)


################################################################################################################
################################################################################################################
################################################################################################################

if __name__ == '__main__':
    batch_train,batch_label=get_batch()
    logits = IGCV3FPN(x=batch_train, num_classes=3, is_training=True).classifier_logits