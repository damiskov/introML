from regressionfuncs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_regression_data():
    """
    Loads data for project 2

    Output:
    - X1, X2: 2 data sets (standardised + one-hot encoded)
    - y: target
    - final_columns: column names (for X2 - leave out last 4 if just working with X1)
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



    X[:, -1] = (X[:, -1] > 0).astype(int)

    # one-hot encoding ca

    N, _ = X.shape

    categories = np.unique(X[:, new_cat_idx])
    encoded_data = np.zeros((N, len(categories)), dtype=int)


    for i in range(N):
        for j, category in enumerate(categories):
            current_cat = int(X[i,new_cat_idx])
            if current_cat == category:
                encoded_data[i,j] = 1
        
    
    X = np.delete(X, new_cat_idx, axis=1)
    X = np.concatenate((X, encoded_data), axis=1)

    final_columns = np.concatenate((continuous_cols, np.array(["num", "ca=0", "ca=1", "ca=2", "ca=3"])))

    N, _ = X.shape
    X[:, new_con_idx] = (X[:, new_con_idx] - np.ones((N, 1))*X[:, new_con_idx].mean(axis=0))/X[:, new_con_idx].std(axis=0)
    y  = (y - np.ones((N))*np.mean(y))/np.std(y)

    
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
    lambdas = np.concatenate((np.power(10.0, range(-5,0), np.arange(2, 100), np.power(10.0, range(1,3)))))

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

def partA_v2(X, y, attributeNames, fname=""):
    """
    Performs part (a) of project 2
    - Performs ridge linear regression on two models, finding optimal lambda.
    - Using 10-fold cross validation, estimates generalisation error for each function according to its lambda.
    - Plots generalisation error vs lambda.
    Inputs:
    - X: data 
    - y: targets
    """

    lambdas = np.concatenate((np.power(10.0, range(-5,0)), np.arange(2, 100), np.power(10.0, range(2, 4))))
    ridge_linear_regression_v2(X, y, lambdas, attributeNames, fname=fname)




if __name__=="__main__":

    #  Load data
    
    X1, X2, y, column_names = load_regression_data()

    # Part a)

    print("------- Regression ---------")
    print("Part a)")
    print("Ridge regression ")

    partA_v2(X1, y, column_names[np.arange(X1.shape[1])], fname="rrX1")
    partA_v2(X2, y, column_names, fname="rrX2")

    print("Part b)")
    print("Comparison of models based on hyperparameter selection")

    h = np.arange(1, 10)
    lambdas = np.concatenate((np.power(10.0, range(-5,0)), np.arange(2, 100), np.power(10.0, range(2, 4))))


    rlr_errors, ANN_errors, baseline_errors, optimal_lambdas, optimal_h = get_hyperparam_generror(X1, y, h, lambdas)
    print(
    f"""
    rlr_errors: {rlr_errors}\n
    ANN_errors: {ANN_errors}\n
    baseline_errors: {baseline_errors}\n
    optimal_lambdas: {optimal_lambdas}\n
    optimal_h: {optimal_h}\n
    """)

    print("----- Statistical Performance Comparison -------")
    
    p_vals = np.zeros((3,3)) # P values
    confidence_intervals = np.empty((3,3), dtype=tuple) # 95% Confidence intervals
    thetas = np.zeros((3,3))

    for i, eA in enumerate([rlr_errors, ANN_errors, baseline_errors]):
        for j, eB in enumerate([rlr_errors, ANN_errors, baseline_errors]):
            if i!=j:
                theta_hat, CI, p = ttest_twomodels(eA, eB)
                p_vals[i,j] = p
                confidence_intervals[i, j] = (np.round(CI[0], 3), np.round(CI[1], 3))
                thetas[i,j] = theta_hat

    print(f"""
    p_vals:\n{p_vals}

    confidence_intervals:\n{confidence_intervals}

    theta_hat:\n{thetas}
    """)









    

