from keras import backend as K
from keras.layers import Conv1D, DepthwiseConv1D, Add, Activation, BatchNormalization
from models.Attentions import SEBlock1D, SABlock1D


def relu6(x):
    return K.relu(x, max_value=6.0)


def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def return_activation(x, nl):
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)
    return x


def conv_block(inputs, filters, kernel, strides, nl):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv1D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return return_activation(x, nl)


def ImprovedBottleneck(inputs, filters, kernel, e, s, attention, nl, num, groups=8, alpha=1.0):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    tchannel = int(e)
    cchannel = int(alpha * filters)
    r = s == 1 and input_shape[-1] == filters
    x = conv_block(inputs, tchannel, 1, 1, nl)
    x = DepthwiseConv1D(kernel, strides=s, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)
    if attention == 'SE':
        x = SEBlock1D(x, ratio=16, num=num)
    elif attention == 'SA':
        x = SABlock1D(x, groups=groups, num=num)
    else:
        raise ValueError('Unsupported attention type!')
    x = Conv1D(cchannel, 1, strides=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    if r:
        x = Add()([x, inputs])
    return x
