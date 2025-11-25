from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import pandas as pd
import csv

# load data using surprise tools instead of the given method
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
trainset, testset = train_test_split(data, test_size=0.2)


param_grid = {
    'n_factors': [150], #[50, 100, 150]
    'n_epochs': [50],
    'biased': [True], #[True, False]
    'init_mean': [-0.1], #[0.1, -0.1]
    'init_std_dev': [0.05], #[0.05, 0.1, 0.2]
    'lr_all': [0.01], #[0.005, 0.01, 0.02]
    'reg_all': [0.1], #[0.02, 0.05, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3, n_jobs=-1)
gs.fit(data)
print("Best MAE score attained: ", gs.best_score['mae'])
print("Best parameters: ", gs.best_params['mae'])


# code to set up our test set using the masked dataset
maskedSet = []
with open('data_movie_lens_100k/ratings_masked_leaderboard_set.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        maskedSet.append((row['user_id'], row['item_id'], 0.0))

model = gs.best_estimator['mae']
model.fit(trainset)
predictions = model.test(maskedSet)

outputfile = "predicted_ratings_leaderboard.txt"

with open(outputfile, 'w') as f:
    for prediction in predictions:
        f.write(f"{round(prediction.est)}\n")