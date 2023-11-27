"""
The two kinds of attention mechanisms for 1D Conv: the keras implementation of the SE-Net and SA-Net
Attention mechanisms for 2D Conv: the keras implementation of the ECA-Net
"""
import math
import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras import layers
from models.GroupNorm import GroupNormalization

"""1D SE-Net used in original BottleNeck block of MobileNetV3"""


def SEBlock1D(input_feature, num, ratio=16):
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        x_input = Permute((2, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        x_input = input_feature
    squeeze = GlobalAveragePooling1D()(x_input)
    excitation = Dense(units=channel // ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=channel)(excitation)
    excitation = Activation('hard_sigmoid')(excitation)
    excitation = Reshape((1, channel))(excitation)
    scale = Multiply(name='SEBlock1D-' + str(num))([x_input, excitation])
    return scale


"""1D SA-Net used in improved BottleNeck block"""


def channel_split(x, groups=4):
    batch, width, filters = x.shape.as_list()[:]
    channels_per_group = filters // groups
    x = K.reshape(x, (-1, width, channels_per_group))
    x1, x2 = layers.Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(x)
    return x1, x2


def channel_shuffle(x, num, groups=2):
    width, filters = x.shape.as_list()[1:]
    channels_per_group = filters // groups
    x = layers.Reshape((width, groups, channels_per_group))(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((width, filters), name='SABlock-' + str(num))(x)
    return x


def channel_attention(inputs_tensor):
    channels = K.int_shape(inputs_tensor)[-1]
    x_global_avg_pool = layers.GlobalAveragePooling1D()(inputs_tensor)
    x = Dense(channels, use_bias=True, kernel_initializer='zeros', bias_initializer='ones')(x_global_avg_pool)
    x = layers.Activation('hard_sigmoid')(x)
    x = layers.Reshape((1, channels))(x)
    output = layers.Multiply()([inputs_tensor, x])
    return output


def spatial_attention(inputs_tensor):
    inputs_tensor = layers.Permute((2, 1))(inputs_tensor)
    channels = K.int_shape(inputs_tensor)[-1]
    x_grop_norm = GroupNormalization(groups=channels, axis=-1)(inputs_tensor)
    x_global_avg_pool = layers.GlobalAveragePooling1D()(x_grop_norm)
    x = Dense(channels, use_bias=True, kernel_initializer='zeros', bias_initializer='ones')(x_global_avg_pool)
    x = layers.Activation('hard_sigmoid')(x)
    x = layers.Reshape((1, channels))(x)
    output = layers.Multiply()([inputs_tensor, x])
    output = layers.Permute((2, 1))(output)
    return output


def SABlock1D(x, groups, num):
    width, filters = x.shape.as_list()[1:]
    x1, x2 = channel_split(x, groups=groups)
    cam = channel_attention(x1)
    sam = spatial_attention(x2)
    y = layers.Concatenate()([cam, sam])
    y = K.reshape(y, (-1, width, filters))
    out = channel_shuffle(y, num, groups=2)
    return out


"""2D ECA-Net"""


def eca_block(inputs_tensor=None, num=None, gamma=2, b=1, **kwargs):
    """
    ECA-NET
    :param inputs_tensor: input_tensor.shape=[batchsize,h,w,channels]
    :param num:
    :param gamma:
    :param b:
    :return:
    """
    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=k, padding="same", name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_relu_' + str(num))(x)
    x = Reshape((1, 1, channels))(x)
    output = Multiply()([inputs_tensor, x])

    return output
