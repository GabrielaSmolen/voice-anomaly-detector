import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import SVC
import lightgbm as lgb
from models.baseline import Baseline
from models.lightgbm_interface import LightGBMInterface



columns = ['Zero crossings', 'Centroid AUC', 'Rolloff AUC', 'Centroid mean',
       'Rolloff mean', 'Centroid STD', 'Rolloff STD', 'Centroid p-t-p value',
       'Rolloff p-t-p value', 'MFCC mean 1', 'MFCC mean 2', 'MFCC mean 3',
       'MFCC mean 4', 'MFCC mean 5', 'MFCC mean 6', 'MFCC mean 7',
       'MFCC mean 8', 'MFCC mean 9', 'MFCC mean 10', 'MFCC mean 11',
       'MFCC mean 12', 'MFCC mean 13', 'MFCC delta 1', 'MFCC delta 2',
       'MFCC delta 3', 'MFCC delta 4', 'MFCC delta 5', 'MFCC delta 6',
       'MFCC delta 7', 'MFCC delta 8', 'MFCC delta 9', 'MFCC delta 10',
       'MFCC delta 11', 'MFCC delta 12', 'MFCC delta 13', 'MFCC delta 2 1',
       'MFCC delta 2 2', 'MFCC delta 2 3', 'MFCC delta 2 4',
       'MFCC delta 2 5', 'MFCC delta 2 6', 'MFCC delta 2 7',
       'MFCC delta 2 8', 'MFCC delta 2 9', 'MFCC delta 2 10',
       'MFCC delta 2 11', 'MFCC delta 2 12', 'MFCC delta 2 13']


def train_model(df, params):
    k_fold = StratifiedKFold(n_splits=5)
    splits = k_fold.split(df, df["Label"])

    test_result = []
    train_result = []

    encode_labels = preprocessing.LabelEncoder()
    encode_labels.fit(df["Label"])
    labels = encode_labels.transform(df["Label"])

    for train_index, test_index in splits:

        df = df[columns]
        train_x = df.loc[train_index]
        test_x = df.loc[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]

        # model = Baseline()
        # model = LogisticRegression(C=70, solver='newton-cg')
        # model = RandomForestClassifier(**params)
        # model = SVC(C=5, kernel='linear', gamma='auto', verbose=2)
        model = LightGBMInterface(params=params)

        model.fit(train_x, labels_train)
        y_predict_test = model.predict(test_x)
        y_predict_train = model.predict(train_x)
        score_test = f1_score(labels_test, y_predict_test)
        score_train = f1_score(labels_train, y_predict_train)
        print(classification_report(labels_test, y_predict_test))
        # print('Test score: ', score_test)
        # print('Train score: ', score_train)
        test_result.append(score_test)
        train_result.append(score_train)
        result_test_mean = np.mean(test_result)
        result_train_mean = np.mean(train_result)
    return result_test_mean, result_train_mean, train_x, model


if __name__ == '__main__':
    df = pd.read_csv('data/features.csv')

    # params = {'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_samples': 100}
    # Random Forest params
    params = {'num_iterations': 40, 'num_leaves': 50, 'min_data_in_leaf': 40, 'objective': 'binary'}  # LightGBM params
    # grid_params = {'C': 5, 'kernel': 'linear', 'gamma': 'auto'} # SVM params

    test_result, train_result, train_x, model = train_model(df, params)

    result_test_mean = np.mean(test_result)
    result_test_std = np.std(test_result)
    print(f'Test score: {result_test_mean}')
    print(f'STD: {result_test_std}')

    result_train_mean = np.mean(train_result)
    result_train_std = np.std(train_result)
    print('Train score: ', result_train_mean)
    print('STD: ', result_train_std)

    # feature_names = [train_x.columns for i in range(train_x.shape[1])]
    # importances = model.feature_importances_
    # std = np.std([
    #     tree.feature_importances_ for tree in model.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # plt.show()

    # with open('params.txt', 'a') as file:
    #     file.write(json.dumps(params) + ' ')
    #     file.write('Train score: ' + str(result_train_mean) + ' ')
    #     file.write('Test score: ' + str(result_test_mean) + '\n')
