import pandas as pd
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class HybridDataset(Dataset):
    def __init__(self, df):
        self.user = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.item = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.meta = torch.tensor(df[['age','is_male','release_year','svdpp_pred']].values,
                                 dtype=torch.float32)
        self.rating = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.meta[idx], self.rating[idx]


class HybridModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, u, i, m):
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        return self.mlp(torch.cat([ue, ie, m], dim=1)).squeeze()
        
class LeaderDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.i = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.m = torch.tensor(df[['age','is_male','release_year','svdpp_pred']].values,
                              dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.m[idx]