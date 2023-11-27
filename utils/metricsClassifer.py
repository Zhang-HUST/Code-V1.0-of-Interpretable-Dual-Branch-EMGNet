import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix

"""Evaluation metrics for motion classification task"""




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


class ModelEvaluationIndex:
    def __init__(self,  true_motion_label, predict_result):
        self.trueMotionLabel = true_motion_label
        self.predictResult = predict_result

    def build(self):
        accuracy, precision, recall, specificity, npv, f1 = classIndex(self.trueMotionLabel, self.predictResult)
        print('accuracy：%.3f, precision：%.3f, recall：%.3f, specificity：%.3f, npv：%.3f, f1：%.3f' % (
            accuracy, precision, recall, specificity, npv, f1))
        result = (accuracy, precision, recall, specificity, npv, f1)
        return result

