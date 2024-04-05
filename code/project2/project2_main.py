from regressionfuncs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_regression_data():
    """
    Loads data for project 2

    Output:
    - X: data set
    - continuous_idx: indices of columns in X containing continuous values
    - cat_idx: indices of restecg and num (only two categorical attributes for model 2)
    """

    filepath=r"/Users/davidmiles-skov/Desktop/Academics/Machine Learning/02450 - Introduction to Machine Learning and Data Mining/Project Work/introML/data/processed_cleveland.data"

    col_names = np.array(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang","oldpeak", "slope","ca","thal","num"])

    continuous_idx = np.array([3, 4, 7, 9])
    #cat_idx = np.array([1, 2, 5, 6, 8, 10, 11, 12])
    cat_idx = np.array([11, 13])
    continuous_cols = col_names[continuous_idx]
    cat_cols = col_names[cat_idx]



    cleveland_data = pd.read_csv(filepath,names =col_names, delimiter=",")

    # Cleaning up data

    cleveland_data.thal=cleveland_data.thal.str.replace("?", "NaN")
    cleveland_data.ca=cleveland_data.ca.str.replace("?", "NaN")
    # Removing missing data

    data = np.array(cleveland_data.values, dtype=np.float64)
    missing_idx = np.isnan(data)
    obs_w_missing = np.sum(missing_idx, 1) > 0
    data_remove_missing = data[np.logical_not(obs_w_missing), :]
    y = data_remove_missing[:, 0]

    cols_to_consider = np.concatenate((continuous_idx, cat_idx))
    # X will have continuous columns, followed by two categorical/binary columns

    # new_con_idx, new_cat_idx = np.array([0, 1, 2, 3]), np.array([4])
    # Trying Ca instead of restecg

    new_con_idx, new_cat_idx = np.array([0, 1, 2, 3]), np.array([4])

    X = data_remove_missing[:, cols_to_consider]

    print("X before num converted to binary and restecg one-hot encoded:")
    print(np.concatenate((continuous_cols, cat_cols)))
    print(X[np.arange(10), :])
    print()

    # Converting num to binary

    X[:, -1] = (X[:, -1] > 0).astype(int)

    # one-hot encoding restecg

    N, _ = X.shape

    categories = np.unique(X[:, new_cat_idx])
    encoded_data = np.zeros((N, len(categories)), dtype=int)

    


    # # Fill the one-hot encoded array
    # for i, category in enumerate(categories):
    #     print(f"Rows where restecg = {category}")
    #     indices = np.where(X[:,new_cat_idx] == category)
    #     encoded_data[indices, i] = 1

    # print(encoded_data[0, :])

    for i in range(N):
        for j, category in enumerate(categories):
            current_cat = int(X[i,new_cat_idx])
            if current_cat == category:
                encoded_data[i,j] = 1
        

    

    # Removing original restecg column, and appending the new "encoded_data" columns to X
    
    X = np.delete(X, new_cat_idx, axis=1)
    X = np.concatenate((X, encoded_data), axis=1)


    print("X after one-hot restecg and binarising num")

    # final_columns = np.concatenate((continuous_cols, np.array(["num", "restecg=0", "restecg=1", "restecg=2"])))
    final_columns = np.concatenate((continuous_cols, np.array(["num", "ca=0", "ca=1", "ca=2", "ca=3"])))
    print(final_columns)
    print(X[np.arange(10), :])
    # print(sum(X[:, -1]))

    # Standardising continuous columns

    print()

    print("X after standardising continuous columns")


    N, _ = X.shape
    X[:, new_con_idx] = (X[:, new_con_idx] - np.ones((N, 1))*X[:, new_con_idx].mean(axis=0))/X[:, new_con_idx].std(axis=0)
    
    print(final_columns)
    print(X[np.arange(1), :])

    # Normalising target feature

    # print("y before standardisation:")
    # print(y[np.arange(5)])
    y  = (y - np.ones((N))*np.mean(y))/np.std(y)
    # print("y after standardisation:")
    # print(y[np.arange(5)])
    
    return X[:, new_con_idx], X,  y, final_columns


def partA(X, y):
    """
    Performs part (a) of project 2
    - Performs ridge linear regression on two models, finding optimal lambda.
    - Using 10-fold cross validation, estimates generalisation error for each function according to its lambda.
    - Plots generalisation error vs lambda.
    Inputs:
    - X_m1, X_m2: data for two models

    """
    lambdas = np.power(10.0, np.arange(-8, 8))
    training_err, validation_err, weights = ridge_linear_regression(X, y, lambdas)
    
    # Plotting error rates

    plt.semilogx(lambdas, training_err, ".-", c='b', label=r'$E_{train}$')
    plt.semilogx(lambdas, validation_err, ".-", c='r', label=r'$E_{validation}$')
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    plt.legend()
    plt.show()

    # Plotting average weights
    plt.semilogx(lambdas, weights.T, ".-")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Average Coefficient Value")
    plt.show()







if __name__=="__main__":

    # Set up
    
    X1, X2, y, column_names = load_regression_data()

    # Part a)

    # partA(X1,y)
    # partA(X2, y)
    h = np.arange(1, 6)
    training_errNN, validation_errNN = regression_NN(X2, y, np.arange(1, 6))
    plt.plot(h, training_errNN, ".-", c='b', label=r'$E_{train}$')
    plt.plot(h, validation_errNN, ".-", c='r', label=r'$E_{validation}$')
    plt.xlabel("Hidden Units")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    plt.legend()
    plt.show()





    

