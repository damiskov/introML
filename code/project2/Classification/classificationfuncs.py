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
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    xlabel,
    ylabel,
    tight_layout,
    savefig
)



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


def RLR_single_fold(X_train, y_train, X_test, y_test, opt_lambda):  
    """
    Returns test error after training on single fold
    """  
    
    M = X_train.shape[1]
    train_err, test_err = 0,0

    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # remove bias regularization
    w = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Evaluate test performance
    
    test_error = np.power(y_test - X_test @ w.T, 2).mean(axis=0)

    return test_error


def ANN_single_fold(X_train, y_train, X_test, y_test,opt_h):
    """
    Trains and tests ANN on single fold, returning test error.
    """
    m = X_train.shape[1]
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(m, opt_h),  # M features to n_hidden_units
            torch.nn.ReLU(),  # 1st transfer function,
            torch.nn.Linear(opt_h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
    )

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_test, y_test = torch.Tensor(X_test),  torch.Tensor(y_test)
    
    loss_fn = torch.nn.MSELoss()  # mean-squared-error loss

    n_replicates, max_iter = 1, 10000

    # Train the net on training data
    net, _, _ = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    predicted = net(X_test)
    mse = torch.mean((predicted - y_test)**2).item()
    return mse
    


def ridge_linear_regression_v2(X, y, lambdas, attributeNames, K=10, fname = ""):

    """
    Implements ridge regression

    Inputs:
    - X, y: data and target
    - lambdas: regression parameter values
    - attributeNames: names of attributes in data
    - K: Number of CV folds
    - save_fig: If plots are to be saved
    """

    N, M = X.shape

    # Add offset attribute
    
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    
    attributeNames = np.array(["Offset"] + [i for i in attributeNames])
    
    M = M + 1

    # Cross validation
    CV = model_selection.KFold(K, shuffle=True)

    # Error rates
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))


    # Weights, etc
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))
    w_noreg = np.empty((M, K))

    # Optimal lambdas values for each inner fold

    optimal_lambda = np.empty(K)

    k = 0

    for train_index, test_index in CV.split(X, y):

        # extract training and test set for current CV fold
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
  

        internal_cross_validation = 10 

        (
            opt_val_err,
            opt_lambda,
            mean_w_vs_lambda,
            train_err_vs_lambda,
            test_err_vs_lambda,
        ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Compute mean squared error without using the input data at all

        Error_train_nofeatures[k] = (
            np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        )
        Error_test_nofeatures[k] = (
            np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
        )


        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # Do not regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = (
            np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
        )
        Error_test_rlr[k] = (
            np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
        )

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = (
            np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
        )
        Error_test[k] = (
            np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
        )
        # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
        # m = lm.LinearRegression().fit(X_train, y_train)
        # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

        optimal_lambda[k] = opt_lambda

        # Display the results for the last cross-validation fold
        if k == K - 1:
            if fname!="":
                figure(k, figsize=(12, 8))
                subplot(1, 2, 1)
                semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
                xlabel("Regularization factor")
                ylabel("Mean Coefficient Values")
                grid()
                # You can choose to display the legend, but it's omitted for a cleaner
                # plot, since there are many attributes
                # legend(attributeNames[1:], loc='best')

                subplot(1, 2, 2)
                print(f"---- Optimal $\lambda$ ------\n")
                print("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
                
                
                loglog(
                    lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
                )
                xlabel("Regularization factor")
                ylabel("Squared error (crossvalidation)")
                legend(["Train error", "Validation error"])
                tight_layout()
                # suptitle(r"Mean Coefficient Values and Error rates vs $\lambda$")
                grid()
                savefig(f"/Users/davidmiles-skov/Desktop/Academics/Machine Learning/02450 - Introduction to Machine Learning and Data Mining/Project Work/introML/figures/Regression/{fname}.png", dpi=600)
                show()
        # To inspect the used indices, use these print statements
        # print('Cross validation fold {0}/{1}:'.format(k+1,K))
        # print('Train indices: {0}'.format(train_index))
        # print('Test indices: {0}\n'.format(test_index))

        k += 1

    
    # Display results
    print("Linear regression without feature selection:")
    print("- Training error: {0}".format(Error_train.mean()))
    print("- Test error:     {0}".format(Error_test.mean()))
    print(
        "- R^2 train:     {0}".format(
            (Error_train_nofeatures.sum() - Error_train.sum())
            / Error_train_nofeatures.sum()
        )
    )
    print(
        "- R^2 test:     {0}\n".format(
            (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()
        )
    )
    print("Regularized linear regression:")
    print("- Training error: {0}".format(Error_train_rlr.mean()))
    print("- Test error:     {0}".format(Error_test_rlr.mean()))
    print(
        "- R^2 train:     {0}".format(
            (Error_train_nofeatures.sum() - Error_train_rlr.sum())
            / Error_train_nofeatures.sum()
        )
    )
    print(
        "- R^2 test:     {0}\n".format(
            (Error_test_nofeatures.sum() - Error_test_rlr.sum())
            / Error_test_nofeatures.sum()
        )
    )

    print("Weights in last fold:")
    for m in range(M):
        print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))
    

def NN_regression(X, y, h, K=10):

    """
    Performs regression using ANN

    Inputs:
    - X: Data
    - y: targets
    - h: number of hidden layers

    Outputs:
    - average mse (estimatesd generalisation error)
    """

    N, M = X.shape

    # Parameters for neural network classifier
    n_hidden_units = h  # number of hidden units
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000


    CV = model_selection.KFold(K, shuffle=True)


    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfser function, i.e. "linear output" - regression
    )
    loss_fn = torch.nn.MSELoss()  # mean-squared-error loss
    errors = []  # make a list for storing generalizaition error in each loop

    for k, (train_index, test_index) in enumerate(CV.split(X, y)):

        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(
            model,
            loss_fn,
            X=X_train,
            y=y_train,
            n_replicates=n_replicates,
            max_iter=max_iter,
        )

        print("\n\tBest loss: {}\n".format(final_loss))

        # Determine estimated class labels for test set
        predicted = net(X_test)

        
        # print(f"shape of y_test_est: {predicted.shape}")

        # # Determine errors and errors
        # se = [i**2 for i in (predicted - y_test)]  # squared error
        # # mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        # mse = sum(se)/len(se)
        # errors.append(mse)  # store error rate for current CV fold
        # print(f"se: {se}\nmse: {mse}")

        new_mse = torch.mean((predicted - y_test)**2).item()
        print(f"New MSE: {new_mse}")
        errors.append(new_mse)



    return round((np.mean(errors)), 4)

def baseline(X, y, K=10):
    """
    Baseline model. Uses most common value in training data as output for all test data.
    """
    CV = model_selection.KFold(K, shuffle=True)
    errors = np.zeros(K)

    for i, (train_idx, test_idx) in enumerate(CV.split(X, y)):
        
        y_train, y_test = y[train_idx], y[test_idx]
        y_pred = mode(y_train)
        
        print("MOST COMMON", y_pred)

        errors[i] = np.mean([x**2 for x in y_pred*np.ones(len(y_test))-y_test])

    return np.mean(errors)



def part_b(X, y, h, lambdas, K=10):
    """
    Performs part b of regression for project 2 - Will return all data for one outer fold 

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

    rlr_errors = np.zeros(K)
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


        # Tuning hyperparameters for Neural Network and Ridge Regression models

        (
            _, 
            opt_lambda,
            _,
            _,
            _,
        ) = rlr_validate(X_rr[train_idx, :], y_train, lambdas, 10)


        errors_ANN = np.zeros(len(h))
       
        for i, hidden_units in enumerate(h):
            errors_ANN[i] = NN_regression(X_train, y_train, hidden_units)
        
        E_nn = np.min(errors_ANN)
        h_opt = h[np.argmin(errors_ANN)]

        print(f"Optimal h: {h}")

        # Saving optimal hyperparameters

        optimal_lambdas[k] = opt_lambda
        optimal_h[k] = h_opt

        # We now have optimal hyperparameters, training and testing on outer fold for generalisation errors

        rlr_errors[k] = RLR_single_fold(X_train, y_train, X_test, y_test, opt_lambda)
        ANN_errors[k] = ANN_single_fold(X_train, y_train, X_test, y_test, h_opt)
        baseline_errors[k] = np.mean([x**2 for x in (np.mean(y_train))*np.ones(len(y_test))-y_test])

    return rlr_errors, ANN_errors, baseline_errors, optimal_lambdas, optimal_h
        




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


        loss_fn = torch.nn.MSELoss()
        E_val, E_train = 0, 0

        # Neural network, one hidden layer, h hidden units

        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(m, h),  # M features to n_hidden_units
            torch.nn.ReLU(),  # 1st transfer function,
            torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )

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

        















