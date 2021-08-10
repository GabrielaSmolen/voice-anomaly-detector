import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

df = pd.read_csv('data/features.csv')

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
    log_reg = LogisticRegression(max_iter=2100)
    log_reg.fit(train_x, labels_train)
    y_predict = log_reg.predict(test_x)
    labels_test = labels[test_index]
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(test_x, labels_test)))
    print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(log_reg.score(train_x, labels_train)))
    test_result.append(log_reg.score(test_x, labels_test))
    train_result.append(log_reg.score(train_x, labels_train))


result_test_mean = np.mean(test_result)
result_test_std = np.std(test_result)
print('Accuracy is: ', result_test_mean)
print('STD: ', result_test_std)

result_train_mean = np.mean(train_result)
result_train_std = np.std(train_result)
print('Accuracy is: ', result_train_mean)
print('STD: ', result_train_std)

print()
