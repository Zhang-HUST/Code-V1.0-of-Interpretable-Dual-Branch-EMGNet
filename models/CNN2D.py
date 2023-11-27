from keras.layers import *
from keras.models import Model
from utils.params import *


def CNN2D():
    # Input Layer
    inputs = Input(shape=(256, 4, 1), name='Inputs')
    X = Conv2D(filters=32, kernel_size=(7, 3), strides=(1, 1), padding='same', name='conv1')(inputs)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 1), strides=(4, 1), name='pool1')(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=64, kernel_size=(5, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 1), strides=(4, 1), name='pool2')(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 2), strides=(4, 2), name='pool3')(X)
    X = Dropout(0.3)(X)

    X = Flatten(name='flatten')(X)
    X = Dropout(0.3)(X)
    X = Dense(64, name='fc1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)
    Y = Dense(classes, activation='softmax', name='fc2')(X)

    model = Model(inputs=inputs, outputs=Y, name='CNN2D')

    return model
