import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import backend as K
from keras import metrics

"""Evaluation metrics for motion classification and joint angle prediction tasks"""


def R2_Score(y_true, y_pred):
    return 1 - metrics.mean_squared_error(y_true, y_pred) / K.var(y_true)


def get_specificity_npv(y1, y2):
    MCM = multilabel_confusion_matrix(y1, y2)
    specificity = []
    npv = []
    for i in range(MCM.shape[0]):
        confusion = MCM[i]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        specificity.append(100 * TN / float(TN + FP))  # Sensitivity
        npv.append(100 * TN / float(FN + TN))  # Negative predictive value（NPV)
    Specificity = np.average(specificity)
    Npv = np.average(npv)

    return Specificity, Npv


def classIndex(true_motion_label, predict_motion_label):
    if predict_motion_label.ndim == 1:
        trueLabel = true_motion_label
        predictLabel = predict_motion_label
    else:
        trueLabel = np.argmax(true_motion_label, axis=1)
        predictLabel = np.argmax(predict_motion_label, axis=1)
    accuracy = 100 * accuracy_score(trueLabel, predictLabel)
    precision = 100 * precision_score(trueLabel, predictLabel, average='macro')
    recall = 100 * recall_score(trueLabel, predictLabel, average='macro')
    f1 = 100 * f1_score(trueLabel, predictLabel, average='macro')
    specificity, npv = get_specificity_npv(trueLabel, predictLabel)

    return accuracy, precision, recall, specificity, npv, f1


def predictIndex(true_angle_label, predict_angle_label):
    trueLabel = true_angle_label
    predictLabel = predict_angle_label
    r2 = r2_score(trueLabel, predictLabel)
    mse = mean_squared_error(trueLabel, predictLabel)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trueLabel, predictLabel)

    return r2, mse, rmse, mae


class ModelEvaluationIndex:
    def __init__(self,  true_angle_label, true_motion_label, predict_result):
        self.trueAngleLabel, self.trueMotionLabel = true_angle_label, true_motion_label
        self.predictResult = predict_result
        self.predictAngleLabel, self.predictMotionLabel = (np.array(self.predictResult[0]),
                                                           np.array(self.predictResult[1]))

    def build(self):
        accuracy, precision, recall, specificity, npv, f1 = classIndex(self.trueMotionLabel, self.predictMotionLabel)
        r2, mse, rmse, mae = predictIndex(self.trueAngleLabel, self.predictAngleLabel)
        print('accuracy：%.3f, precision：%.3f, recall：%.3f, specificity：%.3f, npv：%.3f, f1：%.3f' % (
            accuracy, precision, recall, specificity, npv, f1))
        print('r2：%.3f, mse：%.3f, rmse：%.3f, mae：%.3f' % (r2, mse, rmse, mae))
        result = (accuracy, precision, recall, specificity, npv, f1, r2, mse, rmse, mae)

        return result

