from keras.layers import *
from keras.models import Model
from utils.params import *


def VotingCNN1D():
    # Input Layer
    inputs = Input(shape=(256, 4), name='Inputs')
    inputs1 = Reshape(target_shape=(256*4, 1))(inputs)
    conv1_filters = [128, 128, 32, 32]
    conv1_kernels = [5, 5, 5, 5]
    conv1_strides = [2, 2, 2, 2]
    conv2_filters = [64, 64, 64, 64]
    conv2_kernels = [3, 3, 3, 3]
    conv2_strides = [1, 1, 1, 1]
    conv3_filters = [32, 32, 128, 128]
    conv3_kernels = [3, 3, 3, 3]
    conv3_strides = [1, 1, 1, 1]
    pooling_kernels1 = [4, 4, 4, 4]
    pooling_kernels2 = [4, 4, 4, 4]
    pooling_kernels3 = [2, 2, 2, 2]
    fc1_dim = 100
    output = []
    for i in range(len(conv1_filters)):
        X = Conv1D(filters=conv1_filters[i], kernel_size=conv1_kernels[i], strides=conv1_strides[i], padding='valid')(inputs1)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=pooling_kernels1[i])(X)
        X = Conv1D(filters=conv2_filters[i], kernel_size=conv2_kernels[i], strides=conv2_strides[i], padding='valid')(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=pooling_kernels2[i])(X)
        X = Conv1D(filters=conv3_filters[i], kernel_size=conv3_kernels[i], strides=conv3_strides[i], padding='valid')(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=pooling_kernels3[i])(X)
        X = Flatten()(X)
        X = Dense(fc1_dim, activation='relu')(X)
        if i in [1, 3]:
            X = Dropout(0.3)(X)
        Y = Dense(classes, activation='softmax', name='branch_out'+str(i+1))(X)
        output.append(Y)
    model = Model(inputs=inputs, outputs=output, name='VotingCNN1D')

    return model

