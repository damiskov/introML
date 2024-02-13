import numpy as np
import pandas as pd

# Define filename
filename=r"C:\Users\leona\Downloads\heart+disease\processed.cleveland.data"

# Load data
df = pd.read_csv(filename)
raw_data = df.values

#raw data
cols = range(0, 14)
X = raw_data[:, cols]

#replacing missing values with medians
X[X == '?'] = np.nan;
X = X.astype(float)
X = np.transpose(list(map(lambda col: list(map(lambda elem: np.nanmean(col) if str(elem)  == 'nan' else elem ,col)), np.transpose(X))))

#Summary statistics

means = np.around(np.mean(X, axis=0), 2)
medians =np.around(np.median(X, axis=0), 2)
std = np.around(np.std(X, axis = 0), 2)
ten = np.around(np.percentile(X, 10, axis=0), 2)
twentyfive = np.around(np.percentile(X, 25, axis=0), 2)
seventyfive = np.around(np.percentile(X, 75, axis=0), 2)
ninty = np.around(np.percentile(X, 90, axis=0), 2)
mini = X.min(axis=0)
maxi = X.max(axis=0)
var = np.around(np.var(X, axis=0), 2)

#compute correlation and/or covariance
