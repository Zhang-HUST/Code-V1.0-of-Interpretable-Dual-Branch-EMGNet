from keras.layers import *
from keras.models import Model
from tcn import TCN
from models.SincConv import SincConvLayer
from models.ImprovedBottleneck import ImprovedBottleneck
from utils.tools import slices
from utils.params import *

"""This is the implementation of several different Sinc-EMGNets"""
"""Optional model_type parameters include: Sinc-LSTM, Sinc-BiLSTM, Sinc-GRU, Sinc-BiGRU and Sinc-TCN;
   Optional attention parameters include: 'None', 'SE' and 'SA'.
"""


def SincEMGNets(model_type='Sinc-BiLSTM', attention='SA'):
    predict_out_shape, class_out_shape = PredictLength, classes
    modelName = model_type + '-' + attention

    # Input Layer
    inputs = Input(shape=input_shape, name='Inputs')

    # Shallow feature extraction module
    first_conv_block_outs = []
    for i in range(input_shape[-1]):
        X_input = Lambda(slices, arguments={"index": i}, name='Input-' + str(i + 1))(inputs)
        X = SincConvLayer(SincChannel, SincLength, fs, index=str(i + 1),
                          input_shape=(X_input.shape[0], X_input.shape[-1]),
                          name='Sinc-' + str(i + 1))(X_input)
        X = LeakyReLU(alpha=0.3, name='LReLU1-' + str(i + 1))(X)
        X = MaxPooling1D(pool_size=4, name='MPool1-' + str(i + 1))(X)
        X = Dropout(0.2, name='Drop1-' + str(i + 1))(X)
        first_conv_block_outs.append(X)
    X = Concatenate(axis=-1)(first_conv_block_outs)

    # Deep feature extraction module
    X = Conv1D(filters=64, kernel_size=9, strides=1, padding='same', name='Conv2')(X)
    X = LeakyReLU(alpha=0.3, name='LReLU2')(X)
    if attention == 'None':
        pass
    else:
        X = ImprovedBottleneck(X, filters=64, kernel=9, e=128, s=1, attention=attention, nl='HS',
                               num=1, groups=8, alpha=1.0)
        X = LeakyReLU(alpha=0.3, name='LReLU3')(X)
    X = MaxPooling1D(pool_size=3, name='MPool2')(X)
    X = Dropout(0.2, name='Drop2')(X)

    # Different predictors for joint angle prediction
    if model_type == 'Sinc-BiLSTM':
        Y = Bidirectional(LSTM(RNN_filters1, return_sequences=True), name='BiLSTM1')(X)
        Y = Dropout(0.2, name='Drop3')(Y)
        Y = Bidirectional(LSTM(RNN_filters2, return_sequences=False), name='BiLSTM2')(Y)
    elif model_type == 'Sinc-LSTM':
        Y = LSTM(RNN_filters1, return_sequences=True, name='LSTM1')(X)
        Y = Dropout(0.2, name='Drop3')(Y)
        Y = LSTM(RNN_filters2, return_sequences=False, name='LSTM2')(Y)
    elif model_type == 'Sinc-GRU':
        Y = GRU(RNN_filters1, return_sequences=True, name='GRU1')(X)
        Y = Dropout(0.2, name='Drop3')(Y)
        Y = GRU(RNN_filters2, return_sequences=False, name='GRU2')(Y)
    elif model_type == 'Sinc-BiGRU':
        Y = Bidirectional(GRU(RNN_filters1, return_sequences=True), name='BiGRU1')(X)
        Y = Dropout(0.2, name='Drop3')(Y)
        Y = Bidirectional(GRU(RNN_filters2, return_sequences=False), name='BiGRU2')(Y)
    elif model_type == 'Sinc-TCN':
        Y = TCN(nb_filters=TCN_filters1, kernel_size=3, dilations=(1, 2, 4, 8), return_sequences=True, name='TCN-1')(X)
        Y = Dropout(0.2, name='Drop3')(Y)
        Y = TCN(nb_filters=TCN_filters2, kernel_size=3, dilations=(1, 2, 4, 8, 16), return_sequences=False, name='TCN-2')(Y)
    else:
        raise ValueError('Unsupported model_type!')
    Y = Dropout(0.2, name='Drop4')(Y)
    Y = Dense(predict_out_shape, activation='linear', name='PredictOutput')(Y)

    # Classifier for lower limb motion recognition
    Z = GlobalAveragePooling1D(name='GAP')(X)
    Z = Dropout(0.2, name='Drop5')(Z)
    Z = Dense(class_out_shape, activation='softmax', name='ClassOutput')(Z)
    model = Model(inputs=inputs, outputs=[Y, Z], name=modelName)

    return model, modelName
