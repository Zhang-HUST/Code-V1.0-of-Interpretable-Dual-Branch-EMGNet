import os
import time
import datetime
import numpy as np
import pandas as pd
from models.MLModels import GeneralMLModels
from sklearn.model_selection import train_test_split
from utils.loadTrainData import LoadSplitData
from utils.feature_extraction_tools.feature_extraction import emg_feature_extraction
from utils.metricsClassifer import ModelEvaluationIndex
from utils.tools import make_dir
from utils.params import *

"""This is the main function for model training and testing in both Intra-Subject and Inter-Subject scenarios"""

emg_feature_type = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'WAMP']
fea_normalize_method = 'max-abs'
fea_normalize_level = 'rows'


class ModelTrainTest:
    def __init__(self, model_type):
        self.model_type = model_type

    def train(self, target_subject, save_predict_result=True, param_optimization=True, save_model=True):
        task = 'N-N'
        sub_type = 'targetSubjects'
        total_exp_time = number_repetitions
        utils = LoadSplitData(task, target_subject, with_transfer=False)
        (emg_data, angle_data, _, _, _, _, _, _) = utils.trainTestSplit(sub_type)
        labels = utils.targetMotionLabel
        emg_features = emg_feature_extraction(emg_data, emg_feature_type, fea_normalize_method, fea_normalize_level)
        print('emg_features.shape: ', emg_features.shape, 'labels.shape: ', labels.shape,)
        accuracy, precision, recall, specificity, npv, f1, train_time = self.train_repeats(
            data=emg_features, labels=labels, folds=total_exp_time, target_subject=target_subject,
            param_optimization=param_optimization, save_model=save_model)

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

    def train_repeats(self, data, labels, folds, target_subject, param_optimization, save_model):
        accuracy, precision, recall, specificity, npv, f1 = [], [], [], [], [], []
        train_time = []
        for i in range(folds):
            print("\n" + "==========" * 4 + "%s" % datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + "==========" * 4)
            current_exp_time = i + 1
            print("===============第 %d==%d 次训练=================" % (current_exp_time, folds))
            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_ratio,
                                                                random_state=None, stratify=labels)

            model = GeneralMLModels(self.model_type)
            model.init()
            if param_optimization:
                print('Parameter optimization of GridSearchCV: ')
                best_params = model.parameter_optimization(x_train, y_train)
                print('Optimized parameters:', best_params)
                model.set_params(best_params)
            else:
                print('Use the default parameters: ')

            print('Start model training: ')
            start = time.time()
            model.train(x_train, y_train)
            end = time.time()
            trainTime = end - start
            print('Model training completed, training time: %.3f s' % (float(trainTime)))

            save_dir = self.get_resultDir(target_subject)
            model_save_name = os.path.join(save_dir, ''.join(['model_', str(current_exp_time), '.pkl']))
            if save_model:
                print('Save the model after the last training: ')
                model.save(model_save_name)

            print('Start model evaluation: ')
            pre_y_test = model.predict(x_test)

            test_result = ModelEvaluationIndex(y_test, pre_y_test).build()
            accuracy.append(test_result[0])
            precision.append(test_result[1])
            recall.append(test_result[2])
            specificity.append(test_result[3])
            npv.append(test_result[4])
            f1.append(test_result[5])
            train_time.append(trainTime)

        return accuracy, precision, recall, specificity, npv, f1, train_time

    def get_resultDir(self, target_subject):
        resultDir = os.path.join(os.getcwd(), 'results', 'Intra-Subject', self.model_type, target_subject)
        make_dir(resultDir)

        return resultDir
