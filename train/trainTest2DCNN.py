import os
import time
import datetime
import numpy as np
import pandas as pd
from livelossplot import PlotLossesKeras
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from models.CNN2D import CNN2D
from utils.loadTrainData import LoadSplitData
from utils.metricsClassifer import ModelEvaluationIndex
from utils.resultVisualization import train_process_save
from utils.tools import make_dir
from utils.params import *

max_epoch = 200


class ModelTrainTest:
    def __init__(self):
        self.class_out_shape = classes
        self.modelName = 'CNN2D'

    def get_model(self):
        model = CNN2D()
        print('Successfully building model: %s' % self.modelName)
        return model

    def compile_model(self, model):
        print('model.compile：')
        model.compile(optimizer=optimizers.adam_v2.Adam(lr=initial_lr), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, target_subject, train_plot=True, reduce_lr=True, early_stop=True,
              save_predict_result=True):

        total_exp_time = number_repetitions
        epoch = max_epoch

        accuracy, precision, recall, specificity, npv, f1, train_time = self.train_repeats(
            folds=total_exp_time, epoch=epoch,
            target_subject=target_subject, train_plot=train_plot, reduce_lr=reduce_lr, early_stop=early_stop,
            )

        print('Mean and standard deviation of the test results of %d experiments:' % total_exp_time)
        print('accuracy  =  %.3f / std  =  %.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('precision  =  %.3f / std  =  %.3f' % (np.mean(precision), np.std(precision)))
        print('recall  =  %.3f / std  =  %.3f' % (np.mean(recall), np.std(recall)))
        print('specificity  =  %.3f / std  =  %.3f' % (np.mean(specificity), np.std(specificity)))
        print('npv  =  %.3f / std  =  %.3f' % (np.mean(npv), np.std(npv)))
        print('f1  =  %.3f / std  =  %.3f' % (np.mean(f1), np.std(f1)))
        print('train_time  =  %.3f / std  =  %.3f' % (np.mean(train_time), np.std(train_time)))

        save_dir = self.get_resultDir(target_subject)
        if save_predict_result:
            testResultSaveName = os.path.join(save_dir, 'testResults.csv')
            INDEX = [str(i + 1) for i in range(total_exp_time)]
            df = pd.DataFrame({'accuracy': accuracy, 'precision': precision, 'recall': recall,
                               'specificity': specificity, 'npv': npv, 'f1': f1,  'train_time': train_time},
                              index=INDEX)
            df_copy = df.copy()
            df.loc['mean'] = df_copy.mean()
            df.loc['std'] = df_copy.std()
            df = df.round(5)
            df.to_csv(testResultSaveName, index=True)

    def train_repeats(self, folds, epoch, target_subject, train_plot, reduce_lr, early_stop):
        accuracy, precision, recall, specificity, npv, f1 = [], [], [], [], [], []
        train_time = []
        task = 'N-N'
        sub_type = 'targetSubjects'
        for i in range(folds):
            print("\n" + "==========" * 4 + "%s" % datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + "==========" * 4)
            current_exp_time = i + 1
            print("===============第 %d==%d 次训练=================" % (current_exp_time, folds))
            (_, _, emg_x_train, _, motion_y_train, emg_x_test,
             _, motion_y_test) = LoadSplitData(task, target_subject, with_transfer=False).trainTestSplit(sub_type)
            emg_x_train = emg_x_train[:, :, :, np.newaxis]
            emg_x_test = emg_x_test[:, :, :, np.newaxis]

            model = self.get_model()
            model = self.compile_model(model)

            # Callback functions during model training
            callbacks = []
            # Use the callback function to save the model with the lowest loss on the validation set
            save_dir = self.get_resultDir(target_subject)
            model_save_name = os.path.join(save_dir, ''.join(['model_', str(current_exp_time), '.h5']))
            model_check_point = ModelCheckpoint(filepath=model_save_name, monitor='val_loss',
                                                save_best_only=True, save_weights_only=False,
                                                mode='auto', period=1)
            callbacks.append(model_check_point)

            if train_plot:
                plotlosses = PlotLossesKeras()
                callbacks.append(plotlosses)
                verbose = 0
            else:
                # verbose = 2
                verbose = 1
            if reduce_lr:
                reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=15,
                                             mode='auto', epsilon=0.0001, cooldown=0, min_lr=10 ** -5)
                callbacks.append(reduceLR)
            if early_stop:
                earlyStop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto',
                                          min_delta=0.00001)
                callbacks.append(earlyStop)

            print('Start model training: ')
            start = time.time()
            history = model.fit(emg_x_train,  motion_y_train, validation_split=val_split, epochs=epoch,
                                batch_size=batch_size, shuffle=True, verbose=verbose, callbacks=callbacks)
            end = time.time()
            trainTime = end - start
            print('Model training completed, training time: %.2f min' % (float(trainTime) / 60.0))
            print('Start model evaluation: ')

            best_model = self.get_model()
            best_model.load_weights(model_save_name)
            testPredictResult = best_model.predict(emg_x_test, verbose=1)

            test_result = ModelEvaluationIndex(motion_y_test, testPredictResult).build()
            accuracy.append(test_result[0])
            precision.append(test_result[1])
            recall.append(test_result[2])
            specificity.append(test_result[3])
            npv.append(test_result[4])
            f1.append(test_result[5])
            train_time.append(trainTime)

        return accuracy, precision, recall, specificity, npv, f1, train_time

    def get_resultDir(self, target_subject):

        resultDir = os.path.join(os.getcwd(), 'results', 'Intra-Subject', self.modelName, target_subject)
        make_dir(resultDir)

        return resultDir
