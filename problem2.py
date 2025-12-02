import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from surprise import SVDpp, Dataset as SurpriseDataset, Reader
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

#import data
ratings = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")
movie_info = pd.read_csv("data_movie_lens_100k/movie_info.csv")
user_info = pd.read_csv("data_movie_lens_100k/user_info.csv")

ratings = ratings.merge(movie_info, on="item_id", how="left")
ratings = ratings.merge(user_info, on="user_id", how="left")

#split on user id
gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(ratings, groups=ratings["user_id"]))

train_df = ratings.iloc[train_idx].reset_index(drop=True)
val_df = ratings.iloc[val_idx].reset_index(drop=True)

reader = Reader(line_format="user item rating", sep=",", rating_scale=(1,5), skip_lines=1)
surprise_data = SurpriseDataset.load_from_df(
    train_df[["user_id","item_id","rating"]], reader
)
trainset = surprise_data.build_full_trainset()

svdpp = SVDpp(
    n_factors=50, 
    n_epochs=20, 
    lr_all=5e-3, 
    reg_all=5e-2
)

print("Training SVD++...")
svdpp.fit(trainset)

def svdpp_predict(u, i):
    try:
        return svdpp.predict(int(u), int(i)).est
    except:
        return 3.0 #give average rating in case it fails

train_df["svdpp_pred"] = [
    svdpp_predict(u, i) for u, i in zip(train_df["user_id"], train_df["item_id"])
]
val_df["svdpp_pred"] = [
    svdpp_predict(u, i) for u, i in zip(val_df["user_id"], val_df["item_id"])
]

#calculate residuals
train_df["residual"] = train_df["rating"] - train_df["svdpp_pred"]
val_df["residual"] = val_df["rating"] - val_df["svdpp_pred"]

feature_cols = ["release_year", "age", "is_male"]
train_X = torch.tensor(train_df[feature_cols].fillna(0).values, dtype=torch.float32)
val_X   = torch.tensor(val_df[feature_cols].fillna(0).values, dtype=torch.float32)

train_y = torch.tensor(train_df["residual"].values, dtype=torch.float32).view(-1,1)
val_y   = torch.tensor(val_df["residual"].values, dtype=torch.float32).view(-1,1)

class ResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ResidualDataset(train_X, train_y), batch_size=256, shuffle=True)
val_loader   = DataLoader(ResidualDataset(val_X, val_y), batch_size=256, shuffle=False)

class ResidualNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

nn_model = ResidualNN(len(feature_cols))
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss() #this is mae loss

train_mae_list = []
val_mae_list = []

def compute_mae(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x)
            preds.append(p)
            trues.append(y)
    preds = torch.cat(preds).squeeze()
    trues = torch.cat(trues).squeeze()
    return torch.mean(torch.abs(preds - trues)).item()


print("\nTraining Residual NN...")
epochs = 20

search_space = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "hidden_sizes": [
        [64, 32],
        [128, 64],
        [128, 64, 32],
        [256, 128]
    ],
    "batch_size": [128, 256, 512],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3]
}

def build_model(hidden_sizes, dropout):
    layers = []
    in_dim = len(feature_cols)
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)

def run_trial(hparams):
    # Build model
    model = build_model(
        hidden_sizes=hparams["hidden_sizes"],
        dropout=hparams["dropout"]
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"]
    )

    trial_loader = DataLoader(
        ResidualDataset(train_X, train_y),
        batch_size=hparams["batch_size"],
        shuffle=True
    )

    for _ in range(5):
        model.train()
        for x, y in trial_loader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

    # Evaluate
    model.eval()
    preds = model(val_X).detach().numpy().flatten()
    mae = np.mean(np.abs(preds - val_df["residual"].values))
    return mae, model

N_TRIALS = 20
best_trial_mae = float("inf")
best_hparams = None
best_model = None

for t in range(N_TRIALS):
    # Sample a random configuration
    hparams = {
        k: random.choice(v)
        for k, v in search_space.items()
    }

    mae, model = run_trial(hparams)

    print(f"Trial {t+1}/{N_TRIALS}")
    print("Params:", hparams)
    print(f"MAE = {mae:.6f}\n")

    if mae < best_trial_mae:
        best_trial_mae = mae
        best_hparams = hparams
        best_model = model

print("\n🎉 Best Hyperparameters Found:")
print(best_hparams)
print(f"Best Val MAE = {best_trial_mae:.6f}\n")

print("\nTraining Residual NN with best hyperparameters...")

nn_model = build_model(
    hidden_sizes=best_hparams["hidden_sizes"],
    dropout=best_hparams["dropout"]
)

optimizer = torch.optim.Adam(
    nn_model.parameters(),
    lr=best_hparams["lr"],
    weight_decay=best_hparams["weight_decay"]
)

train_loader = DataLoader(
    ResidualDataset(train_X, train_y),
    batch_size=best_hparams["batch_size"],
    shuffle=True
)


for epoch in range(1, epochs+1):
    nn_model.train()
    train_losses = []

    for x, y in train_loader:
        optimizer.zero_grad()
        pred = nn_model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_mae = compute_mae(nn_model, train_loader)
    val_mae   = compute_mae(nn_model, val_loader)

    train_mae_list.append(train_mae)
    val_mae_list.append(val_mae)

    print(f"Epoch {epoch} | Train MAE: {train_mae:.6f} | Val MAE (residuals): {val_mae:.6f}")

def evaluate_beta(beta):
    final_preds = val_df["svdpp_pred"].values + beta * nn_model(val_X).detach().numpy().flatten()
    return np.mean(np.abs(final_preds - val_df["rating"].values))

best_beta, best_mae = None, float("inf")

for beta in [i/20 for i in range(0, 21)]:  # 0.00 ... 1.00
    mae = evaluate_beta(beta)
    print(f"β = {beta:.2f} → MAE = {mae:.6f}")
    if mae < best_mae:
        best_mae = mae
        best_beta = beta

print(f"\nBest β = {best_beta} with MAE = {best_mae:.6f}")

leaderboard = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")

leaderboard = leaderboard.merge(movie_info, on="item_id", how="left")
leaderboard = leaderboard.merge(user_info, on="user_id", how="left")

leader_X = torch.tensor(
    leaderboard[feature_cols].fillna(0).values, dtype=torch.float32
)

svdpp_preds = np.array([
    svdpp_predict(u, i) for u, i in zip(leaderboard["user_id"], leaderboard["item_id"])
])
nn_preds = nn_model(leader_X).detach().numpy().flatten()

final_preds = svdpp_preds + best_beta * nn_preds

with open("predicted_ratings_leaderboard_set.txt", "w") as f:
    for p in final_preds:
        f.write(f"{p}\n")


plt.figure(figsize=(8,5))
plt.plot(train_mae_list, label="Train MAE")
plt.plot(val_mae_list, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("mae_plot.png")
print("\nSaved MAE plot as mae_plot.png")
plt.close()
