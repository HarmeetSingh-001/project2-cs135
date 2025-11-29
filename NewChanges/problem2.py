from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split 
from surprise.model_selection import GridSearchCV
import pandas as pd
import csv
import random
import numpy as np
from surprise import SVDpp
import matplotlib.pyplot as plt
from analysis_tools import ModelAnalysis
from user_splitter import UserDisjointSplitter

# LOAD DATASET AND SPLIT INTO TRAIN/TEST SETS USING USER-DISJOINT SPLITTER
df = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")

splitter = UserDisjointSplitter(val_ratio=0.2, random_state=42)
train_df, val_df = splitter.split(df)
train_df = train_df.rename(columns={"user_id": "user", "item_id": "item"})
val_df = val_df.rename(columns={"user_id": "user", "item_id": "item"})

reader = Reader(rating_scale=(1,5))

train_data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)
val_data = Dataset.load_from_df(val_df[["user", "item", "rating"]], reader)

trainset = train_data.build_full_trainset()
valset = val_data.build_full_trainset().build_testset()


# HYPERPARAMETER TUNING USING GRID SEARCH CV

param_grid = {
    'n_factors': [20],
    'n_epochs': [40],
    'reg_all': [0.05],
    'lr_all': [0.005]  # fixed to save time
}


# EVALUATE ALL MODELS FROM GRID SEARCH
train_mae_list = []
val_mae_list = []
params_list = []

# MANUAL GRID SEARCH IMPLEMENTATION
for n_factors in param_grid["n_factors"]:
    for n_epochs in param_grid["n_epochs"]:
        for lr_all in param_grid["lr_all"]:
            for reg_all in param_grid["reg_all"]:

                params = {
                    "n_factors": n_factors,
                    "n_epochs": n_epochs,
                    "lr_all": lr_all,
                    "reg_all": reg_all
                }


                algo = SVDpp(**params)
                algo.fit(trainset)

                # TRAIN SET MAE
                train_preds = algo.test(trainset.build_testset())
                train_mae = accuracy.mae(train_preds, verbose=False)

                # VALIDATION SET MAE
                val_preds = algo.test(valset)
                val_mae = accuracy.mae(val_preds, verbose=False)

                train_mae_list.append(train_mae)
                val_mae_list.append(val_mae)
                params_list.append(params)

                print(f"  → Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}\n")

best_idx = int(np.argmin(val_mae_list))
best_params = params_list[best_idx]
best_val_mae = val_mae_list[best_idx]
print(f"Best hyperparameters: {best_params} with Val MAE: {best_val_mae:.4f}")  


# USED FOR VISUALIZATION OF HYPERPARAMETERS for SVDpp
# UNCOMMENT AS NEEDED
ModelAnalysis.plot_hyperparameter(param_grid['n_factors'],train_mae_list,val_mae_list,'n_factors')
# ModelAnalysis.plot_hyperparameter(param_grid['n_epochs'],train_mae_list,val_mae_list,'n_epochs')
# ModelAnalysis.plot_hyperparameter(param_grid['reg_all'],train_mae_list,val_mae_list,'reg_all')
# ModelAnalysis.plot_hyperparameter(param_grid['lr_all'],train_mae_list,val_mae_list,'lr_all')


# SETUP MASKED DATASET TO PREDICT FINAL RAITINGS
# maskedSet = []
# with open('data_movie_lens_100k/ratings_masked_leaderboard_set.csv', 'r') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         maskedSet.append((row['user_id'], row['item_id'], 0.0))

# trainset_full = data.build_full_trainset()
# best_model = gs.best_estimator['mae']
# best_model.fit(trainset_full)
# predictions = best_model.test(maskedSet)
        
# outputfile = "predicted_ratings_leaderboard.txt"

# with open(outputfile, 'w') as f:
#     for prediction in predictions:
#         f.write(f"{round(prediction.est)}\n")