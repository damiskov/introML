from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import numpy as np
import torch
from dtuimldmtools import train_neural_net, rlr_validate
import pandas as pd
import numpy as np
from statistics import mode
import sklearn.linear_model as lm
import scipy.stats as st
      

def part_b(X, y, h, lambdas, K=10):
    """
    Performs part b of classification for project 2 - Will return all data for one outer fold 

    Inputs:
    - X: data (one of the outer folds)
    - y: targets (belonging to the outer fold)
    - h: hidden units to be tested for ANN
    - lambdas: regularisation parameters to be tested for RR
    - attributeNames: has to be passed for RR function

    Outputs:
    - E_rr, E_nn, E_baseline: Lowest test error rates for linear ridge regression, ANN and baseline models.
    - h_opt: Optimal number of hidden units in NN
    - lambda_opt: optimal regularisation parameter.
    """

    RLogR_errors = np.zeros(K)
    ANN_errors = np.zeros(K)
    baseline_errors = np.zeros(K)
    optimal_lambdas = np.zeros(K)
    optimal_h = np.zeros(K)
    
    # Add offset attribute for linear regression model
    X_rr = np.concatenate((np.ones((X.shape[0], 1)), X), 1) 


    CV = model_selection.KFold(K, shuffle=True)

    for k, (train_idx, test_idx) in enumerate(CV.split(X, y)):

        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]

        # Tuning hyperparameters for Neural Network and (Logistic) Ridge Regression Model

        opt_lambda = RLogR_opt_lambda(X_train, y_train, lambdas)

        print(f"Optimal lambda: {opt_lambda}")
        
        errors_ANN = np.zeros(len(h))
       
        for i, hidden_units in enumerate(h):
            print(f"Retrieving error for h = {hidden_units}")
            errors_ANN[i] = ANN_opt_h(X_train, y_train, hidden_units)
        
        E_nn = np.min(errors_ANN)
        h_opt = h[np.argmin(errors_ANN)]

        # Saving optimal hyperparameters

        optimal_lambdas[k] = opt_lambda
        optimal_h[k] = h_opt

        # We now have optimal hyperparameters, training and testing on outer fold for generalisation errors

        RLogR_errors[k] = RLogR_single_fold(X_train, y_train, X_test, y_test, opt_lambda)
        ANN_errors[k] = ANN_single_fold(X_train, y_train, X_test, y_test, h_opt)
        baseline_errors[k] = baseline(X_train, y_train, X_test, y_test)

    return RLogR_errors, ANN_errors, baseline_errors, optimal_lambdas, optimal_h
        

"""
Functions using K-fold cross validation to optimise hyperparameters
"""

def RLogR_opt_lambda(X, y, lambdas, K=10):
    """
    Returns optimal regularisation parameter
    
    Inputs:
    - X Nxm, data
    - y, Nx1, target
    - lambdas, array like, list of regularisation parameters to select from
    Outputs:
    - opt_lambda, regularisation parameter corresponding to minimum test error
    """

    CV = model_selection.KFold(K, shuffle=True)

    for train_idx, test_idx in CV.split(X, y):

        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]

        # Standardize the training and set set based on training set mean and std
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        # Fit regularized logistic regression model to training data 

        train_error_rate = np.zeros(len(lambdas))
        test_error_rate = np.zeros(len(lambdas))
        # coefficient_norm = np.zeros(len(lambdas))

        for k in range(0, len(lambdas)):

            mdl = lm.LogisticRegression(penalty="l2", C=1 / lambdas[k])

            mdl.fit(X_train, y_train)

            y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_test).T

            train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
            test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

            # w_est = mdl.coef_[0]
            # coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambdas[opt_lambda_idx]
    
    return opt_lambda


def ANN_opt_h(X, y, h, K=10):

    """
    Performs classification using ANN for a given number of hidden units

    Inputs:
    - X: Data
    - y: targets
    - h: number of hidden layers

    Outputs:
    - average error rate (estimated generalisation error)
    """

    N, M = X.shape

    # Parameters for neural network classifier
    n_hidden_units = h  # number of hidden units
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000


    CV = model_selection.KFold(K, shuffle=True)


    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
        torch.nn.Sigmoid(),  # final transfer function - want binary output
    )

    # Since we're training a neural network for binary classification, we use a
    # binary cross entropy loss (see the help(train_neural_net) for more on
    # the loss_fn input to the function)
    loss_fn = torch.nn.BCELoss()
    errors = np.zeros(K)  # make a list for storing generalizaition error in each loop

    for k, (train_index, test_index) in enumerate(CV.split(X, y)):

        print("\nANN inner Crossvalidation fold: {0}/{1}".format(k + 1, K))
        # Extract training and test set for current CV fold, convert to tensors
        # Converting to torch tensors:

        X_train, y_train = torch.Tensor(X[train_index, :]), torch.Tensor(y[train_index])
        X_test, y_test = torch.Tensor(X[test_index, :]), torch.Tensor(y[test_index])

        # Train the net on training data
        net, final_loss, _ = train_neural_net(
            model,
            loss_fn,
            X=X_train,
            y=y_train,
            n_replicates=n_replicates,
            max_iter=max_iter,
        )

        print("\n\tBest loss: {}\n".format(final_loss))

        # Determine estimated class labels for test set

        y_sigmoid = net(X_test) 
        predicted = (y_sigmoid > 0.5).type(
        dtype=torch.uint8
        ).squeeze()
        actual = y_test.type(dtype=torch.uint8)
        print(f"""
        Shape of predicted values: {predicted.shape}
        Shape of actual values: {actual.shape}

""")
        e = predicted != actual
        error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()

        print(f"Error rate: {error_rate}")

        errors[k] = error_rate


    return np.mean(errors)

"""
Functions for evaluating performance on single fold
"""


def ANN_single_fold(X_train, y_train, X_test, y_test,opt_h):
    """
    Trains and tests ANN on single fold, returning test error.
    """
    N, M = X_train.shape

    # Converting to torch tensors:

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

    # Parameters for neural network classifier
    n_hidden_units = opt_h  # number of hidden units
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000

    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
        torch.nn.Sigmoid(),  # final transfer function - want binary output
    )
    
    loss_fn = torch.nn.BCELoss()

    # Train the net on training data
    net, _, _ = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    # Determine estimated class labels for test set

    y_sigmoid = net(X_test) 
    predicted = (y_sigmoid > 0.5).type(
    dtype=torch.uint8
    ).squeeze()
    actual = y_test.type(dtype=torch.uint8)

    # Calculating error rate

    e = predicted != actual
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()

    return error_rate

def RLogR_single_fold(X_train, y_train, X_test, y_test, l):
    """
    Calculates test error for ridge logistic regression model with 
    regularisation parameter l and trained on X_train
    """

    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    mdl = lm.LogisticRegression(penalty="l2", C=1 / l)

    mdl.fit(X_train, y_train)

    y_test_est = mdl.predict(X_test).T

    err = np.sum(y_test_est != y_test) / len(y_test)

    return err

def baseline(X_train, y_train, X_test, y_test):
    """
    Baseline model. Uses most common value in training data as output for all test data.
    """

    unique_elements, counts = np.unique(y_train, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common_element = unique_elements[most_common_index]

    predicted = np.ones(len(y_test))*most_common_element
    
    err = np.sum(predicted != y_test) / len(y_test)

    return err


"""
Ttest
"""

def ttest_twomodels(mseA, mseB, alpha=0.05):
    """
    Perform a two-sample t-test on the predicted values of two models.

    Parameters:
    - y_true: array-like, true values
    - yhatA: array-like, predicted values of model A
    - yhatB: array-like, predicted values of model B
    - alpha: float, significance level (default=0.05)
    - loss_norm_p: int, norm order for loss calculation (default=1)

    Returns:
    - mean_diff: float, mean difference between the predicted values of model A and model B
    - confidence_interval: tuple, confidence interval of the mean difference
    - p_value: float, p-value of the null hypothesis that the mean difference is zero
    """
    # zA = np.abs(y_true - yhatA) ** loss_norm_p
    # zB = np.abs(y_true - yhatB) ** loss_norm_p
    z = mseA - mseB
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))
    p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)
    return np.mean(z), CI, p









