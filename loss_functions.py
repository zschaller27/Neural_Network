import numpy as np

class MeanSquaredError():
    def error(self, y_pred, y_true):
        return np.mean(np.square(y_true - y_pred))