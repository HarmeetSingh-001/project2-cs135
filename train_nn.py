import pandas as pd
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from ensemble_classes import HybridDataset,HybridModel

emb_dim = 32
learningrate = 1e-3
epochs = 12
batchsize = 512
weight_decay = 1e-5

print("Loading data...")

ratings = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")
masked_df = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")
users = pd.read_csv("data_movie_lens_100k/user_info.csv")
movies = pd.read_csv("data_movie_lens_100k/select_movies.csv")

all_items = pd.concat([
    ratings['item_id'],
    masked_df['item_id'],
    movies['item_id']
]).unique()

all_users = ratings['user_id'].unique()

user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
item_to_idx = {i: j for j, i in enumerate(sorted(all_items))}
with open("user_item_mappings.pkl", "wb") as f:
    pickle.dump((user_to_idx, item_to_idx), f)

# Load SVD++ predictions from saved model
with open("svdpp_model.pkl", "rb") as f:
    svdpp = pickle.load(f)

ratings['svdpp_pred'] = ratings.apply(
    lambda r: svdpp.predict(r['user_id'], r['item_id']).est,
    axis=1
)

age_scaler = MinMaxScaler().fit(users[['age']])
year_scaler = MinMaxScaler().fit(movies[['release_year']])

users['age'] = age_scaler.transform(users[['age']])
movies['release_year'] = year_scaler.transform(movies[['release_year']])

# Save scalers
pickle.dump(age_scaler, open("age_scaler.pkl","wb"))
pickle.dump(year_scaler, open("year_scaler.pkl","wb"))

ratings = ratings.merge(users, on='user_id')
ratings = ratings.merge(movies[['item_id','release_year']], on='item_id')

dataset = HybridDataset(ratings)
loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

num_users = len(user_to_idx)
num_items = len(item_to_idx)

model = HybridModel(num_users, num_items,emb_dim)
opt = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
loss_fn = nn.L1Loss()

print("Training hybrid neural model...")

for epoch in range(epochs):
    running = 0
    for u, i, m, y in loader:
        pred = model(u, i, m)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()

    print(f"Epoch {epoch+1}: MAE={running/len(loader):.4f}")

torch.save(model.state_dict(), "nn_model.pt")
print("Saved nn_model.pt, age_scaler.pkl, year_scaler.pkl")
