from keras.layers import *
from keras.models import Model
from utils.params import *
from utils.tools import slices


def LSTMModel():
    # Input Layer
    inputs = Input(shape=(256, 4), name='Inputs')
    # Shallow feature extraction module
    first_conv_block_outs = []
    for i in range(input_shape[-1]):
        X_input = Lambda(slices, arguments={"index": i}, name='Input-' + str(i + 1))(inputs)
        X = Conv1D(filters=32, kernel_size=9, strides=1, padding='same', name='Conv1-' + str(i + 1))(X_input)
        X = ReLU(name='ReLU1-' + str(i + 1))(X)
        X = MaxPooling1D(pool_size=4, name='MPool1-' + str(i + 1))(X)
        first_conv_block_outs.append(X)
    X = Concatenate(axis=-1)(first_conv_block_outs)
    # Deep feature extraction module
    X = Conv1D(filters=32, kernel_size=7, strides=1, padding='same', name='Conv2')(X)
    X = ReLU(name='ReLU2')(X)
    X = MaxPooling1D(pool_size=4, name='MPool2')(X)

    Y = LSTM(32, return_sequences=True, name='LSTM1')(X)
    Y = Dropout(0.2, name='Drop1')(Y)
    Y = LSTM(64, return_sequences=True, name='LSTM2')(Y)
    Y = Dropout(0.2, name='Drop2')(Y)
    Y = LSTM(64, return_sequences=False, name='LSTM3')(Y)
    Y = Dropout(0.2, name='Drop3')(Y)
    Y = Dense(PredictLength, activation='relu')(Y)
    Y = Dropout(0.2, name='Drop4')(Y)
    Y = Dense(PredictLength, activation='linear', name='PredictOutput')(Y)
    model = Model(inputs=inputs, outputs=Y, name='LSTMModel')

    return model
