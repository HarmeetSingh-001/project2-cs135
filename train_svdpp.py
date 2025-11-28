import pickle
from surprise import SVDpp, Dataset, Reader

nfactors = 50
epochs = 20
learningrate = 5e-3
regularization = 5e-2

print("Loading dataset...")
reader = Reader(
    line_format='user item rating',
    sep=',',
    rating_scale=(1,5),
    skip_lines=1 
)
data = Dataset.load_from_file("data_movie_lens_100k/ratings_all_development_set.csv", reader=reader)
trainset = data.build_full_trainset()

print("Training SVD++...")
svdpp = SVDpp(n_factors=nfactors, n_epochs=nfactors, lr_all=learningrate, reg_all=regularization)
svdpp.fit(trainset)

print("Saving SVD++ model to svdpp_model.pkl...")
with open("svdpp_model.pkl", "wb") as f:
    pickle.dump(svdpp, f)

