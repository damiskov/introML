import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from dtuimldmtools import confmatplot, rocplot, train_neural_net
from classificationfuncs import *
import pandas as pd
from sklearn import model_selection

def load_classfication_data():
  # Define filename
  filepath=r"/Users/davidmiles-skov/Desktop/Academics/Machine Learning/02450 - Introduction to Machine Learning and Data Mining/Project Work/introML/data/processed_cleveland.data"


  # Data was missing first row:
  col_names = np.array(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang","oldpeak", "slope","ca","thal","num"])
  df = pd.read_csv(filepath,names =col_names, delimiter=",")
  continuous_idx = np.array([0, 3, 4, 7, 9])


  df.thal=df.thal.str.replace("?", "NaN")
  df.ca=df.ca.str.replace("?", "NaN")


  data = np.array(df.values, dtype=np.float64)


  missing_idx = np.isnan(data)

  obs_w_missing = np.sum(missing_idx, 1) > 0
  data_drop_missing_obs = data[np.logical_not(obs_w_missing), :]

  X = data_drop_missing_obs[:, continuous_idx]


  attributeNames = np.array(["age", "trestbps", "chol","thalach","oldpeak"])
  classNames = ['Healthy', 'Unhealthy']

  N = len(X)
  y = (data_drop_missing_obs[:, -1] > 0).astype(int) # y = binarised num
  

  # Standardising 

  X = (X - np.ones((N, 1)) * np.mean(X, axis=0))/np.std(X, axis=0)

  y=y.squeeze()

  return X, y


def classifcation_tasks(X, y, lambdas, h, K=10):
  """
  Performs Majority of classification section
  - Outer Fold (train, test data)
    - Hyperparameters tuned: Models (ANN and Log reg) have their hyper parameters tuned (h and lambda) via a 10-fold CV on train
    - optimal hyperparams stored.
    - Error rates for models with their optimal hyperparameters calculated and stored. (model on test data)
  """

  RLogR_errors, ANN_errors, baseline_errors = np.zeros(K), np.zeros(K), np.zeros(K)
  optimal_lambdas, optimal_h = np.zeros(K), np.zeros(K)

  CV = model_selection.KFold(K, shuffle=True)

  for k, (train_idx, test_idx) in enumerate(CV.split(X, y)):
     
    print(f"Fold {k}/{K}")

    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    # Tuning hyperparameters

    # Logistic regression...

    opt_lambda = RLogR_opt_lambda(X_train, y_train, lambdas)

    # ANN

    errors_ANN = np.zeros(len(h))
          
    for i, hidden_units in enumerate(h):
        errors_ANN[i] = ANN_opt_h(X_train, y_train, hidden_units)
    
    E_nn = np.min(errors_ANN)
    h_opt = h[np.argmin(errors_ANN)]

    # Saving optimal hyperparameters

    optimal_lambdas[k] = opt_lambda
    optimal_h[k] = h_opt

    # We now have optimal hyperparameters, training and testing on outer fold for generalisation errors
    
    RLogR_errors[k] = RLogR_single_fold(X_train, y_train, X_test, y_test, opt_lambda)
    
    ANN_errors[k] = ANN_single_fold(X_train, y_train, X_test, y_test, h_opt)
    

    baseline_errors[k] = baseline(y_train, y_test)

  return RLogR_errors, ANN_errors, baseline_errors, optimal_lambdas, optimal_h






if __name__ == "__main__":

  X, y = load_classfication_data()

  print("---- Loaded data for classification ------")
  
  
  print(f"X:\n{X[np.arange(4), :]}")
  print(f"y:\n{y[np.arange(4)]}")

  print("------ Tasks -----")

  print("Comparison of models based on hyperparameter selection")

  h = np.arange(1, 10)
  lambdas = np.concatenate((np.power(10.0, range(-5,0)), np.arange(2, 100), np.power(10.0, range(2, 4))))


  RLogR_errors, ANN_errors, baseline_errors, optimal_lambdas, optimal_h = part_b(X, y, h, lambdas)
  print(
  f"""
  RLogR_errors: {RLogR_errors}\n
  ANN_errors: {ANN_errors}\n
  baseline_errors: {baseline_errors}\n
  optimal_lambdas: {optimal_lambdas}\n
  optimal_h: {optimal_h}\n
  """)

  print("----- Statistical Performance Comparison -------")
    
  p_vals = np.zeros((3,3)) # P values
  confidence_intervals = np.empty((3,3), dtype=tuple) # 95% Confidence intervals
  thetas = np.zeros((3,3))

  for i, eA in enumerate([RLogR_errors, ANN_errors, baseline_errors]):
      for j, eB in enumerate([RLogR_errors, ANN_errors, baseline_errors]):
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

