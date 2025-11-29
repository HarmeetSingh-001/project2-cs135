import pickle
import pandas as pd
from surprise import SVDpp, Dataset, Reader
from user_splitter import UserDisjointSplitter

# REUSED CODE FROM problem2.py TO TRAIN AND SAVE SVD++ MODEL USING SPLITTER
print("Loading dataset...")
df = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")
splitter = UserDisjointSplitter(val_ratio=0.2, random_state=42)
train_df, val_df = splitter.split(df)
train_df = train_df.rename(columns={"user_id": "user", "item_id": "item"})
val_df = val_df.rename(columns={"user_id": "user", "item_id": "item"})

reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)
trainset = train_data.build_full_trainset()


# SELECTED FROM problem2.py HYPERPARAMETER TUNING RESULTS
nfactors = 20
epochs = 40
learningrate = 0.05
regularization = 0.005


print("Training SVD++...")
svdpp = SVDpp(n_factors=nfactors, n_epochs=nfactors, lr_all=learningrate, reg_all=regularization)
svdpp.fit(trainset)

print("Saving SVD++ model to svdpp_model.pkl...")
with open("svdpp_model.pkl", "wb") as f:
    pickle.dump(svdpp, f)

