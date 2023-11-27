import os
import time
import datetime
import numpy as np
import pandas as pd
from livelossplot import PlotLossesKeras
from keras import optimizers
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from models.CNNEMGNets import CNNEMGNets
from models.SincEMGNets import SincEMGNets
from utils.loadTrainData import LoadSplitData
from utils.metrics import ModelEvaluationIndex
from utils.resultVisualization import PlotEvaluationResult, train_process_save
from utils.tools import make_dir
from utils.params import *

"""This is the main function for model training and testing in both Intra-Subject and Inter-Subject scenarios"""


class ModelTrainTest:
    def __init__(self, model_type, attention):
        self.input_shape, self.predict_out_shape, self.class_out_shape = input_shape, PredictLength, classes
        self.modelType, self.attention, self.modelName = model_type, attention, model_type + '-' + attention
        self.losses, self.loss_weights, self.evaluationMetrics = model_compile_settings()

    # For Intra-Subject scenario and source-model of Inter-Subject scenario
    def get_model(self):
        if self.modelType in ['CNN-LSTM', 'CNN-BiLSTM', 'CNN-GRU', 'CNN-BiGRU', 'CNN-TCN']:
            model, _ = CNNEMGNets(self.modelType, self.attention)
        elif self.modelType in ['Sinc-LSTM', 'Sinc-BiLSTM', 'Sinc-GRU', 'Sinc-BiGRU', 'Sinc-TCN']:
            model, _ = SincEMGNets(self.modelType, self.attention)
        else:
            raise ValueError('Unsupported modelType!')
        print('Successfully building model: %s' % self.modelName)

        return model

    # For Intra-Subject scenario and source-model of Inter-Subject scenario
    def compile_model(self, model):
        print('model.compile：')
        model.compile(optimizer=optimizers.adam_v2.Adam(lr=initial_lr), loss=self.losses,
                      loss_weights=self.loss_weights, metrics=self.evaluationMetrics)

        return model

    # For transfer learning of target-model in Inter-Subject scenario
    def get_transferModel(self, task, target_subject, source_model_dir, compile_model):
        pretrained_model = self.get_model()
        if compile_model:
            # Load the pre-trained source domain model weights
            if task == 'N-A':
                souModelName = os.path.join(source_model_dir, 'model_1.h5')
            else:
                souModelName = os.path.join(source_model_dir, target_subject, 'model_1.h5')
            # sourceModel = load_model(souModelName, custom_objects={'R2_Score': R2_Score,'SincConvLayer':SincConvLayer})
            pretrained_model.load_weights(souModelName)
        # Get the outputs of the input layer and branches 1 and 2
        input_layer = pretrained_model.input
        branch1_output = pretrained_model.get_layer(self.modelType.split('-')[-1]+'2').output
        branch2_output = pretrained_model.get_layer('GAP').output
        # Add new network layers to branches 1 and 2
        fc_layer1 = Dense(PredictLength, name='new_added_predict_fc')(branch1_output)
        LeakyReLU_layer1 = LeakyReLU(alpha=0.3, name='new_added_predict_LeakyReLU')(fc_layer1)
        dropout_layer1 = Dropout(0.2, name='new_added_predict_dropout')(LeakyReLU_layer1)
        output1 = Dense(PredictLength, activation='linear', name='PredictOutput')(dropout_layer1)

        fc_layer2 = Dense(32, name='new_added_class_fc')(branch2_output)
        LeakyReLU_layer2 = LeakyReLU(alpha=0.3, name='new_added_class_LeakyReLU')(fc_layer2)
        dropout_layer2 = Dropout(0.2, name='new_added_class_dropout')(LeakyReLU_layer2)
        output2 = Dense(classes, activation='softmax', name='ClassOutput')(dropout_layer2)
        model = Model(inputs=input_layer, outputs=[output1, output2], name=self.modelName)
        if compile_model:
            # Compile the model and set different initial learning rates for different network layers
            new_added_layers = ['new_added_predict_fc', 'new_added_predict_LeakyReLU',
                                'new_added_predict_dropout', 'PredictOutput',
                                'new_added_class_fc', 'new_added_class_LeakyReLU',
                                'new_added_class_dropout', 'ClassOutput']
            model = self.compile_transferModel(model, new_added_layers)

        return model

    # For transfer learning of target-model in Inter-Subject scenario
    def compile_transferModel(self, model, new_added_layers):
        print('model.compile：')
        optimizer = optimizers.adam_v2.Adam()
        for layer in model.layers:
            if layer is not None:
                if layer.name not in new_added_layers:
                    # print('Set a lower learning rate for the original layers in the model')
                    # Set a lower learning rate for the original layers in the model
                    layer_optimizer = optimizers.adam_v2.Adam(learning_rate=initial_tl_lr_low)
                else:
                    # Set a higher learning rate for the new added network layers
                    # print('Set a higher learning rate for the new added network layers')
                    layer_optimizer = optimizers.adam_v2.Adam(learning_rate=initial_tl_lr_high)
                for weight in layer.trainable_weights:
                    optimizer.add_slot(weight, 'lr', layer_optimizer.lr)

        model.compile(optimizer=optimizer, loss=self.losses,
                      loss_weights=self.loss_weights, metrics=self.evaluationMetrics)

        return model

    def train(self, with_transfer, task, target_subject, sub_type, train_plot=True, reduce_lr=True, early_stop=True,
              save_model_history=True, save_predict_result=True, save_plot_result=True):

        total_exp_time = 1 if (with_transfer and sub_type == 'sourceSubjects') else number_repetitions
        if not with_transfer:
            epoch = max_epoch
        else:
            epoch = source_max_epoch if sub_type == 'sourceSubjects' else target_max_epoch

        accuracy, precision, recall, specificity, npv, f1, r2, mse, rmse, mae, train_time = self.train_repeats(
            with_transfer=with_transfer, sub_type=sub_type, folds=total_exp_time, epoch=epoch, task=task,
            target_subject=target_subject, train_plot=train_plot, reduce_lr=reduce_lr, early_stop=early_stop,
            save_model_history=save_model_history, save_plot_result=save_plot_result)
        if not with_transfer or (with_transfer and sub_type == 'targetSubjects'):
            print('Mean and standard deviation of the test results of %d experiments:' % total_exp_time)
            print('accuracy  =  %.3f / std  =  %.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('precision  =  %.3f / std  =  %.3f' % (np.mean(precision), np.std(precision)))
            print('recall  =  %.3f / std  =  %.3f' % (np.mean(recall), np.std(recall)))
            print('specificity  =  %.3f / std  =  %.3f' % (np.mean(specificity), np.std(specificity)))
            print('npv  =  %.3f / std  =  %.3f' % (np.mean(npv), np.std(npv)))
            print('f1  =  %.3f / std  =  %.3f' % (np.mean(f1), np.std(f1)))
            print('r2  =  %.3f / std  =  %.3f' % (np.mean(r2), np.std(r2)))
            print('mse  =  %.3f / std  =  %.3f' % (np.mean(mse), np.std(mse)))
            print('rmse  =  %.3f / std  =  %.3f' % (np.mean(rmse), np.std(rmse)))
            print('mae  =  %.3f / std  =  %.3f' % (np.mean(mae), np.std(mae)))
            print('train_time  =  %.3f / std  =  %.3f' % (np.mean(train_time), np.std(train_time)))
        else:
            print('Test results of %d experiments:' % total_exp_time)
            print('accuracy  =  %.3f' % accuracy[-1])
            print('precision  =  %.3f' % precision[-1])
            print('recall  =  %.3f' % recall[-1])
            print('specificity  =  %.3f' % specificity[-1])
            print('npv  =  %.3f' % npv[-1])
            print('f1  =  %.3f' % f1[-1])
            print('r2  =  %.3f' % r2[-1])
            print('mse  =  %.3f' % mse[-1])
            print('rmse  =  %.3f' % rmse[-1])
            print('mae  =  %.3f' % mae[-1])
            print('train_time  =  %.3f' % train_time[-1])

        save_dir = self.get_resultDir(with_transfer, task, sub_type, target_subject)
        if save_predict_result:
            testResultSaveName = os.path.join(save_dir, 'testResults.csv')
            INDEX = [str(i + 1) for i in range(total_exp_time)]
            df = pd.DataFrame({'accuracy': accuracy, 'precision': precision, 'recall': recall,
                               'specificity': specificity, 'npv': npv, 'f1': f1, 'r2': r2, 'mse': mse,
                               'rmse': rmse, 'mae': mae, 'train_time': train_time}, index=INDEX)
            df_copy = df.copy()
            df.loc['mean'] = df_copy.mean()
            df.loc['std'] = df_copy.std()
            df = df.round(5)
            df.to_csv(testResultSaveName, index=True)

    def train_repeats(self, with_transfer, sub_type, folds, epoch, task, target_subject, train_plot, reduce_lr,
                      early_stop, save_model_history, save_plot_result):
        accuracy, precision, recall, specificity, npv, f1 = [], [], [], [], [], []
        r2, mse, rmse, mae, train_time = [], [], [], [], []
        for i in range(folds):
            print("\n" + "==========" * 4 + "%s" % datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + "==========" * 4)
            current_exp_time = i + 1
            print("===============第 %d==%d 次训练=================" % (current_exp_time, folds))
            (emg_data, angle_data, emg_x_train, angle_y_train, motion_y_train, emg_x_test,
             angle_y_test, motion_y_test) = LoadSplitData(task, target_subject, with_transfer).trainTestSplit(sub_type)

            source_model_dir = os.path.join(os.getcwd(), 'results', 'Inter-Subject', self.modelName, task,
                                            'sourceSubjects')
            if with_transfer and sub_type == 'targetSubjects':
                model = self.get_transferModel(task, target_subject, source_model_dir, compile_model=True)
            else:
                model = self.get_model()
                model = self.compile_model(model)

            # Callback functions during model training
            callbacks = []
            # Use the callback function to save the model with the lowest loss on the validation set
            save_dir = self.get_resultDir(with_transfer, task, sub_type, target_subject)
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
                reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=40,
                                             mode='auto', epsilon=0.0001, cooldown=0, min_lr=10 ** -5)
                callbacks.append(reduceLR)
            if early_stop:
                earlyStop = EarlyStopping(monitor='val_loss', patience=80, verbose=1, mode='auto',
                                          min_delta=0.00001)
                callbacks.append(earlyStop)

            print('Start model training: ')
            start = time.time()
            history = model.fit(emg_x_train, [angle_y_train, motion_y_train], validation_split=val_split, epochs=epoch,
                                batch_size=batch_size, shuffle=True, verbose=verbose, callbacks=callbacks)
            end = time.time()
            trainTime = end - start
            print('Model training completed, training time: %.2f min' % (float(trainTime) / 60.0))
            print('Start model evaluation: ')
            if with_transfer and sub_type == 'targetSubjects':
                best_model = self.get_transferModel(task, target_subject, source_model_dir, compile_model=True)
            else:
                best_model = self.get_model()
            best_model.load_weights(model_save_name)
            testPredictResult = best_model.predict(emg_x_test, verbose=1)
            tarPredictResult = best_model.predict(emg_data, verbose=1)
            # Saving training history
            history_save_name = os.path.join(save_dir, ''.join(['history_', str(current_exp_time), '.csv']))
            history_save_name_fig = os.path.join(save_dir, ''.join(['history_', str(current_exp_time), '.jpg']))

            if save_model_history:
                # saving model training history (.csv / .jpg)
                df = pd.DataFrame.from_dict(history.history)
                df.insert(loc=0, column='epoch', value=range(1, len(df) + 1))
                df.to_csv(history_save_name, encoding='utf-8', index=False)
                train_process_save(history.history, history_save_name_fig)

            if not with_transfer or (with_transfer and sub_type == 'targetSubjects'):
                if save_plot_result:
                    PlotEvaluationResult(current_exp_time, motion_y_test, testPredictResult,
                                         angle_data, tarPredictResult, save_dir)

            test_result = ModelEvaluationIndex(angle_y_test, motion_y_test, testPredictResult).build()
            accuracy.append(test_result[0])
            precision.append(test_result[1])
            recall.append(test_result[2])
            specificity.append(test_result[3])
            npv.append(test_result[4])
            f1.append(test_result[5])
            r2.append(test_result[6])
            mse.append(test_result[7])
            rmse.append(test_result[8])
            mae.append(test_result[9])
            train_time.append(trainTime)

        return accuracy, precision, recall, specificity, npv, f1, r2, mse, rmse, mae, train_time

    def get_resultDir(self, with_transfer, task, sub_type, target_subject):
        if with_transfer:
            if task == 'N-A' and sub_type == 'sourceSubjects':
                resultDir = os.path.join(os.getcwd(), 'results', 'Inter-Subject', self.modelName, task, sub_type)
            else:
                resultDir = os.path.join(os.getcwd(), 'results', 'Inter-Subject', self.modelName, task, sub_type,
                                         target_subject)
        else:
            resultDir = os.path.join(os.getcwd(), 'results', 'Intra-Subject', self.modelName, target_subject)
        make_dir(resultDir)

        return resultDir
