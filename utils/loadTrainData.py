import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from utils.params import *

"""Load and split the training and test sample sets"""


class LoadSplitData:
    def __init__(self, task, target_subject, with_transfer):
        self.task = task
        self.targetSubject = target_subject
        self.EMGLength = EMGLength
        self.channel = EMGChannel
        self.test_ratio = test_ratio
        self.classes = classes
        self.withTransfer = with_transfer
        if self.withTransfer:
            self.targetData, self.targetMotionLabel, self.sourceData, self.sourceMotionLabel = self.loadData()
            print('targetData:', self.targetData.shape, 'targetMotionLabel:', self.targetMotionLabel.shape)
            print('sourceData:', self.sourceData.shape, 'sourceMotionLabel:', self.sourceMotionLabel.shape)
        else:
            self.targetData, self.targetMotionLabel = self.loadData()
            print('targetData:', self.targetData.shape, 'targetMotionLabel:', self.targetMotionLabel.shape)

    def loadData(self):
        tarSubject = self.targetSubject
        if self.withTransfer:
            tarFile = os.path.join(os.getcwd(), 'trainData', 'Inter-Subject', self.task, 'targetSubjects',
                                   ''.join([tarSubject, 'targetTrainData.npz']))
            if self.task == 'N-A':
                souFile = os.path.join(os.getcwd(), 'trainData', 'Inter-Subject', self.task, 'sourceSubjects',
                                       'sourceTrainData.npz')
            else:
                souFile = os.path.join(os.getcwd(), 'trainData', 'Inter-Subject', self.task, 'sourceSubjects',
                                       ''.join([tarSubject, 'sourceTrainData.npz']))
            # Target domain subject data
            with open(tarFile, 'rb') as f:
                targetData = np.load(f)['data']
                targetMotionLabel = np.load(f)['motionLabel']
            # Source domain subject data
            with open(souFile, 'rb') as f:
                sourceData = np.load(f)['data']
                sourceMotionLabel = np.load(f)['motionLabel']

            return targetData, targetMotionLabel, sourceData, sourceMotionLabel
        else:
            tarFile = os.path.join(os.getcwd(), 'trainData', 'Intra-Subject',
                                   ''.join([tarSubject, 'targetTrainData.npz']))
            # Target domain subject data
            with open(tarFile, 'rb') as f:
                targetData = np.load(f)['data']
                targetMotionLabel = np.load(f)['motionLabel']

            return targetData, targetMotionLabel

    def trainTestSplit(self, sub_type):
        if self.withTransfer and sub_type == 'sourceSubjects':
            data, label = self.sourceData, self.sourceMotionLabel
        else:
            data, label = self.targetData, self.targetMotionLabel
        Label = to_categorical(label, num_classes=self.classes)
        x_train, x_test, motion_y_train, motion_y_test = train_test_split(data, Label, test_size=self.test_ratio,
                                                                          random_state=None, stratify=Label)
        emg_data, angle_data = data[:, 0:self.EMGLength, 0:self.channel], data[:, self.EMGLength:, -1]
        emg_x_train, angle_y_train = x_train[:, 0:self.EMGLength, 0:self.channel], x_train[:, self.EMGLength:, -1]
        emg_x_test, angle_y_test = x_test[:, 0:self.EMGLength, 0:self.channel], x_test[:, self.EMGLength:, -1]

        return (emg_data, angle_data, emg_x_train, angle_y_train, motion_y_train,
                emg_x_test, angle_y_test, motion_y_test)
