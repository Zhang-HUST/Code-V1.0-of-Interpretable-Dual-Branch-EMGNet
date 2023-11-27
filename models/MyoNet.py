from keras.layers import *
from keras.models import Model
from utils.tools import slices
from utils.params import *

input_shape = (256, 4)


def MyoNet():
    # Input Layer
    inputs = Input(shape=input_shape, name='Inputs')

    # Shallow feature extraction module
    first_conv_block_outs = []
    for i in range(input_shape[-1]):
        X_input = Lambda(slices, arguments={"index": i}, name='Input-' + str(i + 1))(inputs)
        X = Conv1D(filters=20, kernel_size=11, strides=1, padding='same', name='Conv1-' + str(i + 1))(X_input)
        X = ReLU(name='ReLU1-' + str(i + 1))(X)
        X = MaxPooling1D(pool_size=4, name='MPool1-' + str(i + 1))(X)
        first_conv_block_outs.append(X)
    X = Concatenate(axis=-1)(first_conv_block_outs)

    # Deep feature extraction module
    X = Conv1D(filters=20, kernel_size=9, strides=1, padding='same', name='Conv2')(X)
    X = ReLU(name='ReLU2')(X)
    X = MaxPooling1D(pool_size=4, name='MPool2')(X)

    # Predictors for joint angle prediction
    Y = LSTM(32, return_sequences=True, name='LSTM1')(X)
    Y = LSTM(64, return_sequences=False, name='LSTM2')(Y)
    Y = Dense(PredictLength, activation='linear', name='PredictOutput')(Y)

    # Classifier for lower limb motion recognition
    Z = Flatten(name='Flatten')(X)
    Z = Dense(classes, activation='softmax', name='ClassOutput')(Z)
    model = Model(inputs=inputs, outputs=[Y, Z], name='MyoNet')

    return model
