import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from surprise import SVDpp, Dataset as SurpriseDataset, Reader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys

# INITIAL SETUP
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# DATA LOADING AND MERGE
try:
    ratings = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")
    movie_info = pd.read_csv("data_movie_lens_100k/movie_info.csv")
    user_info = pd.read_csv("data_movie_lens_100k/user_info.csv")
    leaderboard_path = "data_movie_lens_100k/ratings_masked_leaderboard_set.csv"
except FileNotFoundError:
    print("Error: Data files not found.")
    sys.exit(1)

ratings = ratings.merge(movie_info, on="item_id", how="left")
ratings = ratings.merge(user_info, on="user_id", how="left")

# DATA SPLIT
gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=SEED)
train_idx, val_idx = next(gss.split(ratings, groups=ratings["user_id"]))

train_df = ratings.iloc[train_idx].reset_index(drop=True)
val_df = ratings.iloc[val_idx].reset_index(drop=True)

reader = Reader(line_format="user item rating", sep=",", rating_scale=(1,5), skip_lines=1)
surprise_data = SurpriseDataset.load_from_df(
    train_df[["user_id","item_id","rating"]], reader
)
trainset = surprise_data.build_full_trainset()

# SVD++ TRAINING AND HYPERPARAMETER SEARCH
svdpp_search_space = {
    "n_factors": [20, 50, 80],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1],
    "n_epochs": [15, 25]
}

def train_and_eval_svdpp(params):
    model = SVDpp(
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        lr_all=params["lr_all"],
        reg_all=params["reg_all"],
        random_state=SEED,
        verbose=False
    )
    model.fit(trainset)

    val_preds = []
    for u, i in zip(val_df["user_id"], val_df["item_id"]):
        try:
            pred = model.predict(int(u), int(i)).est
        except ValueError: 
            pred = 3.0
        val_preds.append(pred)

    mae = np.mean(np.abs(val_df["rating"].values - np.array(val_preds)))
    return mae, model

best_svdpp = None
best_svdpp_mae = float("inf")
best_svdpp_params = {} # Store the best parameters dictionary
N_SVD_TRIALS = 30

for t in range(N_SVD_TRIALS):
    params = {k: random.choice(v) for k, v in svdpp_search_space.items()}
    mae, model = train_and_eval_svdpp(params)
    if mae < best_svdpp_mae:
        best_svdpp_mae = mae
        best_svdpp = model
        best_svdpp_params = params # Save the parameters

svdpp = best_svdpp

def svdpp_predict(u, i):
    try:
        return svdpp.predict(int(u), int(i)).est
    except ValueError:
        return 3.0

# SVD++ PREDICTIONS AND RESIDUAL CALCULATION
train_df["svdpp_pred"] = [
    svdpp_predict(u, i) for u, i in zip(train_df["user_id"], train_df["item_id"])
]
val_df["svdpp_pred"] = [
    svdpp_predict(u, i) for u, i in zip(val_df["user_id"], val_df["item_id"])
]

train_df["residual"] = train_df["rating"] - train_df["svdpp_pred"]
val_df["residual"] = val_df["rating"] - val_df["svdpp_pred"]

# FEATURE ENGINEERING AND SCALER SETUP
N_FACTORS = best_svdpp.n_factors
BASE_FEATURES = ["release_year", "age", "is_male", "svdpp_pred"]

# Initialize and fit scaler on training data features
scaler = StandardScaler()
scaler.fit(train_df[BASE_FEATURES].fillna(0).values)


# NEW: Pre-calculates the aggregated implicit user preference vector (Z_u)
def calculate_implicit_user_factors(svdpp_model, df, trainset):
    # Map inner_iid to item implicit factors Yj
    inner_iid_to_yj = {
        trainset.to_raw_iid(i): svdpp_model.yj[i] 
        for i in range(len(svdpp_model.yj))
    }
    
    user_implicit_factors = {}
    
    # Iterate through users to calculate Z_u = (1 / sqrt(|R^u|)) * Sum(Yj for j in R^u)
    for user_id, group in df.groupby('user_id'):
        rated_item_ids = group['item_id'].unique()
        Ru_count = len(rated_item_ids)
        
        if Ru_count == 0:
            user_implicit_factors[user_id] = np.zeros(svdpp_model.n_factors)
            continue

        sum_Yj = np.zeros(svdpp_model.n_factors)
        
        for item_id in rated_item_ids:
            if item_id in inner_iid_to_yj:
                sum_Yj += inner_iid_to_yj[item_id]

        # Normalize the sum
        Zu = sum_Yj / np.sqrt(Ru_count)
        user_implicit_factors[user_id] = Zu
        
    return user_implicit_factors

# Calculate implicit user factors for the training set to use in feature extraction
train_implicit_user_factors = calculate_implicit_user_factors(svdpp, train_df, trainset)

def extract_svdpp_features(df, model, trainset, scaler, implicit_user_factors_map):
    # Corrected features: Pu (Explicit User), Qi (Explicit Item), Zu (Implicit User Aggregate)
    pu_list, qi_list, Zu_list = [], [], []
    uid_map = trainset.to_inner_uid
    iid_map = trainset.to_inner_iid

    for u_raw, i_raw in zip(df["user_id"], df["item_id"]):
        # 1. Pu (Explicit User Factor)
        try:
            u_inner = uid_map(u_raw)
            pu = model.pu[u_inner]
        except ValueError:
            pu = np.zeros(N_FACTORS)

        # 2. Qi (Explicit Item Factor)
        try:
            i_inner = iid_map(i_raw)
            qi = model.qi[i_inner]
        except ValueError:
            qi = np.zeros(N_FACTORS)

        # 3. Zu (Implicit User Factor - Aggregate of all Yj)
        # We use the pre-calculated map to get Z_u for the raw user ID
        Zu = implicit_user_factors_map.get(u_raw, np.zeros(N_FACTORS))
        
        pu_list.append(pu)
        qi_list.append(qi)
        Zu_list.append(Zu)
        
    # Combine latent features: Pu, Qi, Zu
    latent_features = np.hstack([np.array(pu_list), np.array(qi_list), np.array(Zu_list)])
    
    base_features_unscaled = df[BASE_FEATURES].fillna(0).values
    base_features_scaled = scaler.transform(base_features_unscaled)

    full_features = np.hstack([latent_features, base_features_scaled])
    
    return full_features

# PYTORCH DATA PREPARATION
# Pass the correct implicit user factors map to the feature extraction function
train_X_full = extract_svdpp_features(train_df, svdpp, trainset, scaler, train_implicit_user_factors)
val_X_full = extract_svdpp_features(val_df, svdpp, trainset, scaler, train_implicit_user_factors)

train_X = torch.tensor(train_X_full, dtype=torch.float32)
val_X = torch.tensor(val_X_full, dtype=torch.float32)

train_y = torch.tensor(train_df["residual"].values, dtype=torch.float32).view(-1,1)
val_y = torch.tensor(val_df["residual"].values, dtype=torch.float32).view(-1,1)

IN_DIM = train_X.shape[1]

# PYTORCH DATASET AND MODEL DEFINITION
class ResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualNN(nn.Module):
    def __init__(self, in_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        current_dim = in_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h
            
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

loss_fn = nn.L1Loss()

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
    
    if preds.numel() == 0:
        return 0.0
    if preds.ndim == 0: 
        return torch.abs(preds - trues).item()
        
    return torch.mean(torch.abs(preds - trues)).item()

# NN HYPERPARAMETER SEARCH
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

def run_trial(hparams):
    model = ResidualNN(
        in_dim=IN_DIM,
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

    TRIAL_EPOCHS = 10
    for _ in range(TRIAL_EPOCHS):
        model.train()
        for x, y in trial_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(val_X).detach().numpy().flatten()
    mae = np.mean(np.abs(preds - val_df["residual"].values))
    return mae, model

best_trial_mae = float("inf")
best_hparams = None
N_TRIALS = 20

for t in range(N_TRIALS):
    hparams = {k: random.choice(v) for k, v in search_space.items()}
    mae, model = run_trial(hparams)
    if mae < best_trial_mae:
        best_trial_mae, best_hparams = mae, hparams

# FINAL NN TRAINING
nn_model = ResidualNN(
    in_dim=IN_DIM,
    hidden_sizes=best_hparams["hidden_sizes"],
    dropout=best_hparams["dropout"]
)

optimizer = torch.optim.Adam(
    nn_model.parameters(),
    lr=best_hparams["lr"],
    weight_decay=best_hparams["weight_decay"]
)

epochs = 50 
train_mae_list, val_mae_list = [], []
best_val_mae = float('inf')
patience_counter = 0
MAX_PATIENCE = 10

final_train_loader = DataLoader(
    ResidualDataset(train_X, train_y),
    batch_size=best_hparams["batch_size"],
    shuffle=True
)
train_loader_base = DataLoader(ResidualDataset(train_X, train_y), batch_size=256, shuffle=True)
val_loader_base = DataLoader(ResidualDataset(val_X, val_y), batch_size=256, shuffle=False)


for epoch in range(1, epochs + 1):
    nn_model.train()
    for x, y in final_train_loader:
        optimizer.zero_grad()
        loss = loss_fn(nn_model(x), y)
        loss.backward()
        optimizer.step()

    train_mae = compute_mae(nn_model, train_loader_base)
    val_mae = compute_mae(nn_model, val_loader_base)

    train_mae_list.append(train_mae)
    val_mae_list.append(val_mae)
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= MAX_PATIENCE:
            break

# ENSEMBLE BLENDING WEIGHT OPTIMIZATION
def evaluate_beta(beta):
    nn_model.eval()
    with torch.no_grad():
        nn_preds = nn_model(val_X).detach().numpy().flatten()
    
    final_preds = val_df["svdpp_pred"].values + beta * nn_preds
    return np.mean(np.abs(final_preds - val_df["rating"].values))

best_beta, best_mae = None, float("inf")

for beta in [i/20 for i in range(21)]:
    mae = evaluate_beta(beta)
    if mae < best_mae:
        best_mae = mae
        best_beta = beta

# LEADERBOARD PREDICTION AND OUTPUT
try:
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard = leaderboard.merge(movie_info, on="item_id", how="left")
    leaderboard = leaderboard.merge(user_info, on="user_id", how="left")
except FileNotFoundError:
    sys.exit(1)

leaderboard["svdpp_pred"] = [
    svdpp_predict(u, i) for u, i in zip(leaderboard["user_id"], leaderboard["item_id"])
]
svdpp_preds = leaderboard["svdpp_pred"].values 

# The implicit user factors must be recalculated for the entire dataset (train + validation) to ensure all users are covered, 
# especially if the leaderboard contains users only present in the validation set.
# However, to strictly follow the train/test split principle (and since the SVD++ model was only trained on train_df),
# we will use the factors calculated from the training set.
leaderboard_implicit_user_factors = calculate_implicit_user_factors(svdpp, leaderboard, trainset)

# Pass the correct implicit user factors map to the feature extraction function
leader_X_full = extract_svdpp_features(leaderboard, svdpp, trainset, scaler, leaderboard_implicit_user_factors)

leader_X = torch.tensor(leader_X_full, dtype=torch.float32)

nn_model.eval()
with torch.no_grad():
    nn_preds = nn_model(leader_X).detach().numpy().flatten()

final_preds = svdpp_preds + best_beta * nn_preds

output_filename = "predicted_ratings_leaderboard_set.txt"
with open(output_filename, "w") as f:
    for p in final_preds:
        clipped_p = np.clip(p, 1.0, 5.0)
        f.write(f"{clipped_p}\n")

# HYPERPARAMETER AND RESULT SUMMARY
print("--- Hybrid Model Training Summary ---")
print(f"SVD++ Best Hyperparameters:")
print(f"  - Factors (k): {best_svdpp_params['n_factors']}")
print(f"  - Learning Rate (lr_all): {best_svdpp_params['lr_all']}")
print(f"  - Regularization (reg_all): {best_svdpp_params['reg_all']}")
print(f"  - Epochs (n_epochs): {best_svdpp_params['n_epochs']}")
print(f"Residual NN Best Hyperparameters:")
print(f"  - Learning Rate (lr): {best_hparams['lr']}")
print(f"  - Dropout: {best_hparams['dropout']}")
print(f"  - Hidden Layers: {best_hparams['hidden_sizes']}")
print(f"  - Batch Size: {best_hparams['batch_size']}")
print(f"  - Weight Decay: {best_hparams['weight_decay']}")
print(f"Ensemble Blending Weight (Beta): {best_beta:.2f}")
print(f"Final Blended Validation MAE: {best_mae:.4f}")
print("-------------------------------------")

# PLOT MAE CURVE
plt.figure(figsize=(8,5))
plt.plot(train_mae_list, label="Train MAE")
plt.plot(val_mae_list, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.title("Residual NN MAE vs Epoch (Latent Factors Included)")
plt.grid(True)
plt.savefig("mae_plot.png")
plt.close()