from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import numpy as np
import torch
from dtuimldmtools import dbplotf, train_neural_net, visualize_decision_boundary
import pandas as pd

def ridge_linear_regression(X, y, lambdas, K=10):
    """
    Simple Linear Regression for model 2 
    Inputs:
    - X: features
    - y: target variable
    - lambdas: list of regularization parameters
    - K: number of folds for cross-validation
    Output:
    - training error, validation error, average_weights
    """
    N, m = X.shape
    training_err, validation_err = np.zeros(len(lambdas)), np.zeros(len(lambdas))
     
    CV = model_selection.KFold(K, shuffle=True)

    weights = np.empty((m, len(lambdas)))


    for i, l in enumerate(lambdas):

        E_val, E_train = 0, 0

        for train_idx, test_idx in CV.split(X, y):

         
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # print(X_train.shape)
            # print(y_train.shape)
            


            # One-hot encode categorical variables
            # Standardise continuous variables


            # Initialising model

            model = Ridge(alpha=l)

            # Fitting the model
            model.fit(X_train, y_train)


            # # Get model weights
            # Xty = X_train.T @ y_train
            # XtX = X_train.T @ X_train

            # weights = np.linalg.solve(XtX, Xty).squeeze()
    
            # Predictions
            train_predict, test_predict  = model.predict(X_train), model.predict(X_test)
    

            # Evaluation (mean squared error)
            E_train += mean_squared_error(y_train, train_predict)
            E_val += mean_squared_error(y_test, test_predict)
            weights[:, i] +=  model.coef_/K

            

        training_err[i], validation_err[i] = E_train/K, E_val/K


    return training_err, validation_err, weights


# def nn_error(predicted, actual):
#     """
#     Simple helper function to calculate MSE
#     """
#     se = (predicted.float() - actual.float()) ** 2  # squared error
#     mse = (sum(se).type(torch.float) / len(actual)).data.numpy()  # mean
#     print(mse.shape)
#     return mse

def nn_error(predicted, actual):
    """
    Calculate mean squared error (MSE) between predicted and actual values.
    """
    # Calculate squared error
    se = (predicted.float() - actual.float()) ** 2

    # Calculate mean squared error (MSE)
    mse = torch.mean(se).item()  # Convert to Python float

    return mse


def regression_NN(X, y, num_hidden, K=10):
    """
    Neural Network for regression
    Inputs:
    - X: features
    - y: target variable
    - num_hidden: list of different number of hidden layers
    - K: number of folds for cross-validation
    Output:
    - training error, validation error
    """
    N, m = X.shape
    CV = model_selection.KFold(K, shuffle=True)
    training_err, validation_err = np.empty(len(num_hidden)), np.empty(len(num_hidden))


    for i, h in enumerate(num_hidden):

        print(f"Training Neural Network with {h} hidden units")

        # Neural network, one hidden layer, h hidden units
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(m, h),  # M features to n_hidden_units
            torch.nn.ReLU(),  # 1st transfer function,
            torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )

        loss_fn = torch.nn.MSELoss()
        E_val, E_train = 0, 0

        for train_idx, test_idx in CV.split(X, y):
            
            X_train = torch.Tensor(X[train_idx, :])
            y_train = torch.Tensor(y[train_idx])
            X_test = torch.Tensor(X[test_idx, :])
            y_test = torch.Tensor(y[test_idx])

             # Train the net on training data
            
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=1,
                max_iter=10000,
            )

           # Calculating error rates
            
            train_predict, test_predict = net(X_train), net(X_test)

        
            E_train += nn_error(train_predict, y_train)
            E_val += nn_error(test_predict, y_test)
        
        
        
        training_err[i] = E_train/K
        validation_err[i] = E_val/K


    return training_err, validation_err

        















