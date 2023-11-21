import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import GridSpec
from utils.params import *
# print(plt.style.available)
plt.style.use('Solarize_Light2')
# font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
font = {'weight': 'normal', 'size': 20}

"""Visualization of motion classification (confusion_matrix) and joint angle prediction results"""


class PlotEvaluationResult:
    def __init__(self, current_exp_time, tar_motion_y_test, test_predict_result, tar_angle_data,
                 tar_predict_result, save_dir):
        self.trueTestMotionLabel, self.predictTestMotionLabel = np.argmax(tar_motion_y_test, axis=1), np.argmax(
            np.array(test_predict_result[1]), axis=1)
        self.trueAngleLabel, self.predictAngleLabel = tar_angle_data, np.array(tar_predict_result[0])
        self.exp_time = current_exp_time
        self.SaveDir = save_dir
        self.cm, self.cm_normalized = None, None
        self.motionLabels = motionLabels
        self.plotResult()

    def plotResult(self):
        fig = plt.figure(figsize=(15, 12), dpi=100, constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        self.cm = confusion_matrix(self.trueTestMotionLabel, self.predictTestMotionLabel)
        self.cm_normalized = 100 * self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        ax1 = fig.add_subplot(gs[0, 0])
        ax1 = sns.heatmap(self.cm, cmap='Blues', cbar=True, annot=True, square=True, annot_kws={'size': 16},
                          xticklabels=self.motionLabels, yticklabels=self.motionLabels)
        ax1.set_xlabel('Predicted type', font)
        ax1.set_ylabel('True type', font)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2 = sns.heatmap(self.cm_normalized, fmt='.3f', cmap='Blues', cbar=True, annot=True, square=True,
                          annot_kws={'size': 16}, xticklabels=self.motionLabels,
                          yticklabels=self.motionLabels)
        ax2.set_xlabel('Predicted type', font)
        ax2.set_ylabel('True type', font)
        # num=int(np.floor(self.trueAngleLabel.shape[0]/4))
        # label1=[]
        # label2=[]
        # for i in range(num*4):
        #     k=i
        #     label1.append(self.trueAngleLabel[k,:])
        #     label2.append(self.predictAngleLabel[k,:])
        # trueAngleLabel,predictAngleLabel=np.array(label1),np.array(label2)
        trueAngleLabel, predictAngleLabel = self.trueAngleLabel, self.predictAngleLabel
        print(trueAngleLabel.shape, predictAngleLabel.shape)
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.plot(trueAngleLabel.reshape(-1), label='True')
        ax3.plot(predictAngleLabel.reshape(-1), 'r', label='Predicted')
        ax3.legend(fontsize=16)
        ax3.set_xlabel('Time (ms)', font)
        ax3.set_ylabel('Angle (^o)', font)
        # fig.tight_layout()
        figSaveName = os.path.join(self.SaveDir, ''.join(['plotResult_', str(self.exp_time), '.jpg']))
        plt.savefig(figSaveName, dpi=300, format='jpg')
        self.saveData()
        plt.show()

    def saveData(self):
        # Saving the class labels on the test set
        save1 = pd.DataFrame({'trueMotion': self.trueTestMotionLabel, 'predictMotion': self.predictTestMotionLabel})
        saveName1 = os.path.join(self.SaveDir, ''.join(['classResults_', str(self.exp_time), '.csv']))
        save1.to_csv(saveName1, index=False)
        # Saving the confusion matrix
        labels = self.motionLabels
        saveName2 = os.path.join(self.SaveDir, ''.join(['confuseMatrices_', str(self.exp_time), '.xlsx']))
        confuse_matrix = pd.DataFrame(self.cm, index=labels, columns=labels)
        normalized_confuse_matrix = pd.DataFrame(self.cm_normalized, index=labels, columns=labels)
        writer = pd.ExcelWriter(saveName2, engine='xlsxwriter')
        confuse_matrix.to_excel(writer, sheet_name='cm', index=True)
        normalized_confuse_matrix.to_excel(writer, sheet_name='normalized_cm', index=True)
        writer.close()
        # saving angle data
        saveName3 = os.path.join(self.SaveDir, ''.join(['predictedResults_', str(self.exp_time), '.csv']))
        save3 = pd.DataFrame({'trueAngle': self.trueAngleLabel.reshape(-1), 'predictAngle': self.predictAngleLabel.reshape(-1)})
        save3.to_csv(saveName3, index=False)


def train_process_save(history, save_name):
    epoch = range(1, len(history['lr'])+1)
    all_columns = [['ClassOutput_accuracy', 'val_ClassOutput_accuracy'],
                   ['PredictOutput_R2_Score', 'val_PredictOutput_R2_Score'],
                   ['ClassOutput_loss', 'val_ClassOutput_loss'],
                   ['PredictOutput_loss', 'val_PredictOutput_loss'],
                   ['loss', 'val_loss']]
    fig = plt.figure(figsize=(12, 15), dpi=300, constrained_layout=True)
    gs = GridSpec(len(all_columns)+1, 1, figure=fig)
    for i, columns in enumerate(all_columns):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(epoch, history[columns[0]], 'b', label=columns[0])
        ax.plot(epoch, history[columns[1]], 'r', label=columns[1])
        ax.legend(fontsize=16)
        ax.set_xlabel('epoch', font)

    ax = fig.add_subplot(gs[len(all_columns), 0])
    ax.plot(epoch, history['lr'], 'r', label='learning rate')
    ax.legend(fontsize=16)
    ax.set_xlabel('epoch', font)

    fig.tight_layout()
    plt.savefig(save_name, dpi=300, format='jpg')
    plt.show()








