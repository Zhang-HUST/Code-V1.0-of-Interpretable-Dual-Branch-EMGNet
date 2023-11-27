from utils.metrics import R2_Score

"""Data related parameters"""
fs = 1000
classes = 3
EMGLength = 256
EMGChannel = 4
PredictLength = 64
motionLabels = ['GAIT', 'SIT', 'STAND']

"""Model architecture related parameters"""
input_shape = (EMGLength, EMGChannel)
SincChannel = 32
SincLength = 64
RNN_filters1 = 32
RNN_filters2 = 64
TCN_filters1 = 32
TCN_filters2 = 64

"""Model training and testing related parameters"""
test_ratio = 0.2
val_split = 0.1
batch_size = 32
initial_lr = 0.001
initial_tl_lr_high = 0.001
initial_tl_lr_low = 0.0001
max_epoch = 2000
source_max_epoch = 1000
target_max_epoch = 500
loss_scale = 1
number_repetitions = 5


def model_compile_settings():
    losses = {'PredictOutput': 'mse', 'ClassOutput': 'categorical_crossentropy'}
    predict_loss_scale = 1.0 * loss_scale
    loss_weights = {'PredictOutput': predict_loss_scale, 'ClassOutput': 1.0}
    evaluation_metrics = {'PredictOutput': [R2_Score, 'mae'], 'ClassOutput': ['accuracy']}

    return losses, loss_weights, evaluation_metrics

