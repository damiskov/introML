import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from dtuimldmtools import confmatplot, rocplot, train_neural_net
import classificationfuncs
import pandas as pd
from sklearn import model_selection




# Define filename
filename=r"C:\Users\leona\Downloads\heart+disease\processed.cleveland.data"

# Data was missing first row:
col_names = np.array(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang","oldpeak", "slope","ca","thal","num"])
df = pd.read_csv(filename,names =col_names, delimiter=",")
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
y = data_drop_missing_obs[:, -1].squeeze()

#for i in range(len(y)):
 #  if y[i] != 0:
  #     y[i] = 1 #unhealthy are 1

        

X = (X - np.ones((N, 1)) * np.mean(X, axis=0))/np.std(X, axis=0)

N, M = X.shape
K = 10
CV = model_selection.KFold(K, shuffle=True)



lambda_interval = np.concatenate((np.power(10.0, range(-5,0)), np.arange(2, 100), np.power(10.0, range(2, 4))))
hidden_units = np.arange(1, 10)

classificationfuncs.part_b(X, y, hidden_units, lambda_interval)