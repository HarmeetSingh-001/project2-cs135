import pandas as pd
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from ensemble_classes import HybridDataset,HybridModel
from itertools import product
from sklearn.model_selection import train_test_split
from user_splitter import UserDisjointSplitter
from analysis_tools import ModelAnalysis



param_grid = {
    'emb_dim': [16,32],
    'learningrate': [1e-1],
    'weight_decay': [1e-3, 1e-4, 1e-5],
    'epochs': [10],
    'batchsize': [256]
}


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

num_users = len(user_to_idx)
num_items = len(item_to_idx)

# use gpu if available, I am not waiting 4 hours
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# make train val split
splitter = UserDisjointSplitter(val_ratio=0.2, random_state=42)
train_df, val_df = splitter.split(ratings)


train_user_index = train_df['user_id'].map(user_to_idx).values
train_item_index = train_df['item_id'].map(item_to_idx).values
train_svdpp = train_df['svdpp_pred'].values
train_ratings = train_df['rating'].values

val_user_index = val_df['user_id'].map(user_to_idx).values
val_item_index = val_df['item_id'].map(item_to_idx).values
val_svdpp = val_df['svdpp_pred'].values
val_ratings = val_df['rating'].values

train_dataset = HybridDataset(pd.DataFrame({
    'user_id': train_user_index,
    'item_id': train_item_index,
    'age': train_df['age'].values,
    'is_male': train_df['is_male'].values,
    'release_year': train_df['release_year'].values,
    'svdpp_pred': train_svdpp,
    'rating': train_ratings
}))

val_dataset = HybridDataset(pd.DataFrame({
    'user_id': val_user_index,
    'item_id': val_item_index,
    'age': val_df['age'].values,
    'is_male': val_df['is_male'].values,
    'release_year': val_df['release_year'].values,
    'svdpp_pred': val_svdpp,
    'rating': val_ratings
}))



# build model using params with helper function
def train_evaluate_model(params):
    emb_dim, lr, wd, batch_size, epochs = params

    # loader but for both train and val instead of just full thing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = HybridModel(num_users, num_items, emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.L1Loss()

    print("Training hybrid neural model...")

    # check for overfitting
    best_val_mae = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # training
        model.train()
        train_running = 0
        for u, i, m, y in train_loader:
            u, i, m, y = u.to(device), i.to(device), m.to(device), y.to(device)
            pred = model(u, i, m)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_running += loss.item()
        train_mae = train_running / len(train_loader)
        

        # validation
        model.eval()
        val_running = 0
        with torch.no_grad():
            for u, i, m, y in val_loader:
                u, i, m, y = u.to(device), i.to(device), m.to(device), y.to(device)
                pred = model(u, i, m)
                loss = loss_fn(pred, y)
                val_running += loss.item()
        val_mae = val_running / len(val_loader)
        print(f"Epoch {epoch+1}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")

        # early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                print("Early stopping triggered.")
                break
    
    mae = val_running / len(val_loader)
    print(f"Epoch {epoch+1}: MAE={mae:.4f}")
    return mae, model, train_mae, val_mae

# do grid search
best_mae = float('inf')
best_model = None
best_params = None
train_mae_list = []
val_mae_list = []

for params in product(param_grid['emb_dim'], param_grid['learningrate'],
                      param_grid['weight_decay'], param_grid['batchsize'],
                      param_grid['epochs']):
    mae, model, train_mae, val_mae = train_evaluate_model(params)
    train_mae_list.append(train_mae)
    val_mae_list.append(val_mae)
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_params = params

# USED FOR VISUALIZATION OF HYPERPARAMETERS for SVDpp
# UNCOMMENT AS NEEDED
# ModelAnalysis.plot_hyperparameter(param_grid['emb_dim'],train_mae_list,val_mae_list,'emb_dim')
# ModelAnalysis.plot_hyperparameter(param_grid['learningrate'],train_mae_list,val_mae_list,'learningrate')
# ModelAnalysis.plot_hyperparameter(param_grid['weight_decay'],train_mae_list,val_mae_list,'weight_decay')
# ModelAnalysis.plot_hyperparameter(param_grid['batchsize'],train_mae_list,val_mae_list,'batchsize')
# ModelAnalysis.plot_hyperparameter(param_grid['epochs'],train_mae_list,val_mae_list,'epochs')

print(f"\nBest MAE: {best_mae:.4f} with params: {best_params}")

# Save best model
torch.save(best_model.state_dict(), "nn_model.pt")
print("Saved nn_model.pt")
print("Saved nn_model.pt, age_scaler.pkl, year_scaler.pkl")
