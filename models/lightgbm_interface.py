import numpy as np
import lightgbm as lgb


class LightGBMInterface:
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, train_x, labels_train):
        dataset = lgb.Dataset(train_x, labels_train)
        self.model = lgb.train(self.params, dataset)
        return self

    def predict(self, data):
        return np.round(self.model.predict(data))
