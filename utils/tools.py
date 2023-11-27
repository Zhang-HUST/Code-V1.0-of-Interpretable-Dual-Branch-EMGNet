"""some common tools"""

import os
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler


def slices(x, index):
    return x[:, :, index:index + 1]


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass


def is_string_in_list(lst, target_string):
    return target_string in lst


def data_normalize(data, normalize_method, normalize_level):
    if normalize_level == 'matrix':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            # print(np.min(scaler.data_min_),np.max(scaler.data_max_))
            normalized_data = ((data - np.min(scaler.data_min_)) / (
                    np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = (data / np.maximum(np.abs(np.max(scaler.data_max_)),
                                                 np.abs(np.min(scaler.data_min_)))) * scaler.scale_
        else:
            raise ValueError('Error: Unsupported normalize_method！')
    elif normalize_level == 'rows':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        else:
            raise ValueError('Error: Unsupported normalize_method！')
    else:
        raise ValueError('Error: Unsupported normalize_level！')

    return normalized_data
