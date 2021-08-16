import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score
from baseline import Baseline
import json


df = pd.read_csv('data/features.csv')

params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 30, 'min_samples_leaf': 50, 'max_samples': 10}

k_fold = StratifiedKFold(n_splits=5)
splits = k_fold.split(df, df["Label"])

test_result = []
train_result = []

for train_index, test_index in splits:
    train_x = df.loc[train_index]
    test_x = df.loc[test_index]
    del train_x["Label"]
    del train_x["ID"]
    del test_x["Label"]
    del test_x["ID"]
    encode_labels = preprocessing.LabelEncoder()
    encode_labels.fit(df["Label"])
    encode_labels.classes_ = list(encode_labels.classes_)
    labels = encode_labels.transform(df["Label"])
    labels_train = labels[train_index]
    # model = Baseline()
    # model = LogisticRegression(C=30, solver='newton-cg')
    model = RandomForestClassifier(**params)
    model.fit(train_x, labels_train)
    y_predict_test = model.predict(test_x)
    y_predict_train = model.predict(train_x)
    labels_test = labels[test_index]
    score_test = accuracy_score(labels_test, y_predict_test)
    score_train = accuracy_score(labels_train, y_predict_train)
    print(classification_report(labels_test, y_predict_test))
    print('Test score: ', score_test)
    print('Train score: ', score_train)
    test_result.append(score_test)
    train_result.append(score_train)


result_test_mean = np.mean(test_result)
result_test_std = np.std(test_result)
print('Accuracy is: ', result_test_mean)
print('STD: ', result_test_std)

result_train_mean = np.mean(train_result)
result_train_std = np.std(train_result)
print('Accuracy is: ', result_train_mean)
print('STD: ', result_train_std)

with open('params.txt', 'a') as file:
    file.write(json.dumps(params) + ' ')
    file.write('Train score: ' + str(result_train_mean) + ' ')
    file.write('Test score: ' + str(result_test_mean) + '\n')
