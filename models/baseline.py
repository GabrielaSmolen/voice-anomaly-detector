import numpy as np


class Baseline:
    def __init__(self):
        self.label = None

    def fit(self, data, labels):
        bincount = np.bincount(labels)
        self.label = np.argmax(bincount)
        return self

    def predict(self, data):
        predictions = np.repeat(self.label, data.shape[0])
        return predictions

