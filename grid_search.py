import pandas as pd
from train import train_model
import itertools
import tqdm
import json

df = pd.read_csv('data/features.csv')

# grid_params = {'C': [10, 5, 30], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['auto', 'scale']}

# grid_params = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30],
#           'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [2, 5, 10, 20],
#           'max_samples': [100, 200, 300]}

grid_params = {'num_iterations': [5, 10, 20, 50], 'num_leaves': [5, 10, 20, 50],
               'objective': ['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie', 'binary', 'cross_entropy', 'cross_entropy_lambda',
                             'lambdarank', 'rank_xendcg', 'multiclass', 'multiclassova']}

best_score = 0

for params in tqdm.tqdm(itertools.product(*grid_params.values())):
    params = {'num_iterations': params[0], 'num_leaves': params[1], 'tree_learner': params[2], 'objective': params[3]}
    # params = {'C': params[0], 'kernel': params[1], 'gamma': params[2]}
    test_result, train_result, train_x, model = train_model(df, params)
    if test_result > best_score:
        best_score = test_result
        best_parameters = params

    with open('grid_search_logs.txt', 'a') as file:
        file.write(json.dumps(params) + '\n')
        file.write('Test score: ' + str(test_result) + ' ')
        file.write('Train score: ' + str(train_result) + '\n')

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
