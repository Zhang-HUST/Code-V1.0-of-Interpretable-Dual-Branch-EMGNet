import tensorflow as tf
from keras.layers import Layer
from keras.utils import conv_utils
import numpy as np


class SincConvLayer(Layer):
    """
    Sinc-based convolution Keras layer

    Parameters
    ----------
    nb_filters : `int`
        Number of filters (= number of output channels).
    kernel_size : `int`
        Convolution filter width/length (will be increased by one if even).
    sample_freq : `int`
        Sample rate of input audio.
    stride : `int`
        Convolution stride param. Defaults to 1.
    padding : `string`
        Convolution padding param. Defaults to "VALID".
    min_low_hz : `int`
        Minimum lowest frequency for pass band filter. We set it to 20.
    min_band_hz : `int`
        Minimum frequency for pass band filter. We set it to 30.
    """

    @staticmethod
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def __init__(self, nb_filters, kernel_size, sample_freq, index, stride=1, padding="VALID", min_low_hz=20,
                 min_band_hz=30, **kwargs):
        super(SincConvLayer, self).__init__(**kwargs)

        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.sample_freq = sample_freq
        self.stride = stride
        self.padding = padding
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.index = str(index)

        # Force filter size to be odd for later optimizations with symmetry
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

    def build(self, input_shape):
        # Set trainable parameters
        self.low_hz = self.add_weight(
            name='low_hz'+self.index,
            shape=(self.nb_filters,),
            initializer="zeros",
            trainable=True)
        self.band_hz = self.add_weight(
            name='band_hz' + self.index,
            shape=(self.nb_filters,),
            initializer="zeros",
            trainable=True)

        # Initialize weights with frequencies of the mel-scale filter-bank
        low_freq_mel = self.hz_to_mel(30)
        high_freq_mel = self.hz_to_mel(self.sample_freq / 2 - (self.min_low_hz + self.min_band_hz))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num=self.nb_filters + 1)
        hz_points = self.mel_to_hz(mel_points)
        self.set_weights([hz_points[:-1], np.diff(hz_points)])

        # Determine half of t
        t_linspace = np.arange(-(self.kernel_size - 1) / 2, 0)
        t = tf.Variable(2 * np.pi * t_linspace / self.sample_freq, name='param3'+self.index)
        t = tf.cast(t, "float32")
        self.t = tf.reshape(t, (1, -1))

        # Determine half of the hamming window
        n = np.linspace(0, (self.kernel_size / 2) - 1, num=int((self.kernel_size / 2)))
        window = 0.54 - 0.46 * tf.cos(2 * np.pi * n / self.kernel_size)
        window = tf.cast(window, "float32")
        self.window = tf.Variable(window, name='param4' + self.index)

    def call(self, X):
        low = self.min_low_hz + tf.abs(self.low_hz)
        high = tf.clip_by_value(low + self.min_band_hz + tf.abs(self.band_hz), self.min_low_hz, self.sample_freq / 2)
        band = high - low

        low_times_t = tf.linalg.matmul(tf.reshape(low, (-1, 1)), self.t)
        high_times_t = tf.linalg.matmul(tf.reshape(high, (-1, 1)), self.t)

        band_pass_left = ((tf.sin(high_times_t) - tf.sin(low_times_t)) / (self.t / 2)) * self.window
        band_pass_center = tf.reshape(2 * band, (-1, 1))
        band_pass_right = tf.reverse(band_pass_left, [1])

        filters = tf.concat([band_pass_left,
                             band_pass_center,
                             band_pass_right], axis=1)
        filters = filters / (2 * band[:, None])

        # TF convolution assumes data is stored as NWC
        filters = tf.transpose(filters)
        filters = tf.reshape(filters, (self.kernel_size, 1, self.nb_filters))
        name5 = 'my_conv'+self.index
        return tf.nn.conv1d(X, filters, self.stride, self.padding,name=name5)

    def get_config(self):
        config = {"kernel_size": self.kernel_size, "nb_filters": self.nb_filters, "sample_freq ": self.sample_freq,
                  "index": self.index,
                  "stride": self.stride, "padding": self.padding, "min_low_hz": self.min_low_hz,
                  "min_band_hz": self.min_band_hz,
                  }
        base_config = super(SincConvLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        out_width_size = conv_utils.conv_output_length(
            input_shape[1],
            self.kernel_size,
            padding="valid",
            stride=1,
            dilation=1)
        return (input_shape[0], out_width_size, self.nb_filters)
