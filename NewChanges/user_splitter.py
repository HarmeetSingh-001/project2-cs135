import pandas as pd
import numpy as np

class UserDisjointSplitter:
    def __init__(self, val_ratio=0.3, random_state=None):
        self.val_ratio = val_ratio
        self.random_state = random_state

    def split(self, df):
        # GET UNIQUE USERS
        users = df["user_id"].unique()
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(users)

        # SPLIT USERS INTO TRAIN AND VALIDATION SETS
        val_size = int(len(users) * self.val_ratio)
        val_users = set(users[:val_size])
        train_users = set(users[val_size:])
        train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
        val_df = df[df["user_id"].isin(val_users)].reset_index(drop=True)

        return train_df, val_df