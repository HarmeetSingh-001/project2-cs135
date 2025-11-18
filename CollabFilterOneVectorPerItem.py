'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
import pandas as pd
import matplotlib.pyplot as plt

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users),
            c_per_item=ag_np.ones(n_items), 
            U= ag_np.array(0.001 * random_state.randn(n_users, self.n_factors)),
            V=ag_np.array(0.001 * random_state.randn(n_items, self.n_factors)), 
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''

        # Use the learned parameters from param_dict if not provided
        if mu is None:
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict['b_per_user']
        if c_per_item is None:
            c_per_item = self.param_dict['c_per_item']
        if U is None:   
            U = self.param_dict['U']
        if V is None:
            V = self.param_dict['V']

        # get parameters for the user/item given
        mu_term = mu[0]
        b_user_term = b_per_user[user_id_N]
        c_item_term = c_per_item[item_id_N]

        # compute dot product then combine all terms to get predictions
        yhat_N = ag_np.sum(U[user_id_N, :] * V[item_id_N, :], axis=1)
        yhat_N = yhat_N + mu_term + b_user_term + c_item_term   
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TIP: use self.alpha to access regularization strength

        # unpack data and call predict method
        user_id_N = ag_np.array(data_tuple[0], dtype=int)
        item_id_N = ag_np.array(data_tuple[1], dtype=int)
        y_N = ag_np.array(data_tuple[2])
        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)

        # calculate squared error term
        residuals = y_N - yhat_N
        squared_error = ag_np.sum(residuals ** 2)

        # take values from param_dict
        U = param_dict['U']
        V = param_dict['V']
        b_per_user = param_dict['b_per_user']
        c_per_item = param_dict['c_per_item']

        # calculate regularization term using parameters from param_dict
        reg_term = self.alpha * (ag_np.sum(U ** 2) + ag_np.sum(V ** 2) + ag_np.sum(b_per_user ** 2) + ag_np.sum(c_per_item ** 2))

        # loss calculation
        loss_total = squared_error + reg_term
        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)

    # Used for 1A
    
    n_factors_list = [2, 10, 50]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, n_factors in enumerate(n_factors_list):
        model = CollabFilterOneVectorPerItem(
            n_epochs=25, batch_size=1000, step_size=.25,
            n_factors=n_factors, alpha=0, random_state=42)
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)

        # Plot training and validation loss curves
        axes[i].plot(model.trace_rmse_train, label='Train Loss')
        axes[i].plot(model.trace_rmse_valid, label='Valid Loss')
        axes[i].set_title(f'CollabFilterOneVectorPerItem (n_factors={n_factors})')
        axes[i].set_xlabel('Recorded Epochs')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Used for 1B
    # alpha_list = [0, 0.001, 0.01, 0.1]
    # results = {} 

    # for alpha in alpha_list:
    #     model = CollabFilterOneVectorPerItem(
    #         n_epochs=200,
    #         batch_size=1000,
    #         step_size=0.25,
    #         n_factors=50,
    #         alpha=alpha,
    #         random_state=42
    #     )
    #     model.init_parameter_dict(n_users, n_items, train_tuple)
    #     model.fit(train_tuple, valid_tuple)

    #     best_val_rmse = min(model.trace_rmse_valid)
    #     results[alpha] = (best_val_rmse, model)

    # best_alpha = min(results, key=lambda a: results[a][0])
    # best_model = results[best_alpha][1]


    # plt.figure(figsize=(7,5))
    # plt.plot(best_model.trace_rmse_train, label='Train Loss')
    # plt.plot(best_model.trace_rmse_valid, label='Valid Loss')
    # plt.title(f'Best Model (alpha={best_alpha}, n_factors={50})')
    # plt.xlabel('Recorded Epochs')
    # plt.ylabel('Loss (RMSE)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

