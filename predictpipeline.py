import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from ensemble_classes import HybridModel,LeaderDataset

alpha = 0.5

print("Loading SVD++...")
with open("svdpp_model.pkl","rb") as f:
    svdpp = pickle.load(f)
    
with open("user_item_mappings.pkl", "rb") as f:
    user_to_idx, item_to_idx = pickle.load(f)

print("Loading neural model...")
users = pd.read_csv("data_movie_lens_100k/user_info.csv")
movies = pd.read_csv("data_movie_lens_100k/select_movies.csv")

age_scaler = pickle.load(open("age_scaler.pkl","rb"))
year_scaler = pickle.load(open("year_scaler.pkl","rb"))

masked = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")

if "rating" in masked.columns:
    masked = masked[['user_id','item_id']]
elif "0.0" in masked.columns:
    masked = masked[['user_id','item_id']]

users['age'] = age_scaler.transform(users[['age']])
movies['release_year'] = year_scaler.transform(movies[['release_year']])

masked = masked.merge(users, on="user_id", how="left")
masked = masked.merge(movies[['item_id','release_year']], on="item_id", how="left")

masked.fillna({
    'age': users['age'].mean(),
    'is_male': 0.5,
    'release_year': movies['release_year'].mean()
}, inplace=True)

masked['svdpp_pred'] = masked.apply(
    lambda r: svdpp.predict(r['user_id'], r['item_id']).est,
    axis=1
)

masked['user_idx'] = masked['user_id'].map(user_to_idx)
masked['item_idx'] = masked['item_id'].map(item_to_idx)
masked['user_idx'] = masked['user_idx'].fillna(0).astype(int)
masked['item_idx'] = masked['item_idx'].fillna(0).astype(int)

n_users = len(user_to_idx)
n_items = len(item_to_idx)

model = HybridModel(n_users, n_items)
model.load_state_dict(torch.load("nn_model.pt"))
model.eval()

loader = DataLoader(LeaderDataset(masked), batch_size=1024, shuffle=False)

nn_preds = []
with torch.no_grad():
    for u,i,m in loader:
        nn_preds.extend(model(u,i,m).numpy())

masked['nn_pred'] = nn_preds


masked['final'] = alpha*masked['svdpp_pred'] + (1-alpha)*masked['nn_pred']
masked['final'] = masked['final'].clip(1,5).round().astype(int)

with open("predicted_ratings_leaderboard.txt","w") as f:
    for r in masked['final']:
        f.write(str(r) + "\n")
