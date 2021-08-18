import pandas as pd
from train import train_model
import itertools
import tqdm
import json

df = pd.read_csv('data/features.csv')

grid_params = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30],
          'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [2, 5, 10, 20],
          'max_samples': [100, 200, 300]}

best_score = 0

for params in tqdm.tqdm(itertools.product(*grid_params.values())):
    params = {'n_estimators': params[0], 'max_depth': params[1], 'min_samples_split': params[2],
              'min_samples_leaf': params[3], 'max_samples': params[4]}
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
