import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def predictIndex(true_angle_label, predict_angle_label):
    trueLabel = true_angle_label
    predictLabel = predict_angle_label
    r2 = r2_score(trueLabel, predictLabel)
    mse = mean_squared_error(trueLabel, predictLabel)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trueLabel, predictLabel)

    return r2, mse, rmse, mae


class ModelEvaluationIndex:
    def __init__(self, true_angle_label, predict_result):
        self.trueAngleLabel = true_angle_label
        self.predictAngleLabel = predict_result

    def build(self):
        r2, mse, rmse, mae = predictIndex(self.trueAngleLabel, self.predictAngleLabel)
        print('r2：%.3f, mse：%.3f, rmse：%.3f, mae：%.3f' % (r2, mse, rmse, mae))
        result = (r2, mse, rmse, mae)

        return result
