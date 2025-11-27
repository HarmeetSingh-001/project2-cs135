from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import pandas as pd
import csv
import random
import numpy as np
from surprise import SVDpp
import matplotlib.pyplot as plt
from analysis_tools import ModelAnalysis

# load data using surprise tools instead of the given method
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
trainset, testset = train_test_split(data, test_size=0.2)


param_grid = {
    'n_factors': [20, 50],
    'n_epochs': [20, 40],
    'reg_all': [0.05, 0.1],
    'lr_all': [0.005]  # fixed to save time
}


gs = GridSearchCV(SVDpp, param_grid, measures=['mae'], cv=10, n_jobs=-1)
gs.fit(data)

train_mae_list = []
val_mae_list = []

# for params in gs.cv_results['params']:
#     algo = SVD(**params)
#     algo.fit(trainset)
#     train_preds = algo.test(trainset.build_testset())
#     train_mae = accuracy.mae(train_preds, verbose=False)
#     val_preds = algo.test(testset)
#     val_mae = accuracy.mae(val_preds, verbose=False)
#     train_mae_list.append(train_mae)
#     val_mae_list.append(val_mae)
print("Best MAE score attained: ", gs.best_score['mae'])
print("Best parameters: ", gs.best_params['mae'])

#ModelAnalysis.plot_hyperparameter(param_grid['n_factors'],train_mae_list,val_mae_list,'n_factors')
#ModelAnalysis.plot_hyperparameter(param_grid['n_epochs'],train_mae_list,val_mae_list,'n_epochs')


# code to set up our test set using the masked dataset
maskedSet = []
with open('data_movie_lens_100k/ratings_masked_leaderboard_set.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        maskedSet.append((row['user_id'], row['item_id'], 0.0))

trainset_full = data.build_full_trainset()
best_model = gs.best_estimator['mae']
best_model.fit(trainset_full)
predictions = best_model.test(maskedSet)
        

outputfile = "predicted_ratings_leaderboard.txt"

with open(outputfile, 'w') as f:
    for prediction in predictions:
        f.write(f"{round(prediction.est)}\n")