import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from scipy.linalg import svd
import sys


# Path to data
filepath= "data/processed_cleveland.data"

col_names = np.array(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang","oldpeak", "slope","ca","thal","num"])

continuous_idx = np.array([0, 3, 4, 7, 9])
cat_idx = np.array([1, 2, 5, 6, 8, 10, 11, 12])
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
data_drop_missing_obs = data[np.logical_not(obs_w_missing), :]


# Summary Statistics

# Histograms for continuous data

def histograms(save_figs=False):

    for col_name, col_data in zip(continuous_cols, data_drop_missing_obs[:, continuous_idx].T):

        fig, ax = plt.subplots()
        num_bins = 20

        
        n, bins, patches = ax.hist(col_data, num_bins, color='lightsteelblue', edgecolor="black", alpha=0.5)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title(col_name)

        mean, std, median = np.mean(col_data), np.std(col_data), np.median(col_data)
        ax.axvline(x=mean,color='maroon',linestyle='--',label=r"$\mu$: "+str(round(mean, 2)))
        ax.axvline(x=median, color='forestgreen', linestyle="--", label=f"median: {round(median, 2)}")
        
        fig.legend()
        plt.show()
        
        # FOR SAVING IMAGES, CHANGE PATH IF NECESSARY
        if save_figs:
            path = "figures/Histograms/"+col_name+".png"
            fig.savefig(path, dpi=300)
       

# Bar plots for catgeroical data

def barplots(save_figs=False):

    for col_name, col_data in zip(cat_cols, data_drop_missing_obs[:, cat_idx].T):
        x = np.array(list(set(col_data)))
        y = np.array([np.count_nonzero(col_data==i) for i in x])
        plt.bar(x, y, color='lightcoral', alpha=0.8, edgecolor="black")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title(col_name)
        plt.xticks(x)
        
        if save_figs:
            path = "figures/barcharts/"+col_name+".png"
            plt.savefig(path, dpi=300)
        
        plt.show()


# histograms()
# barplots()
            
# PCA
            
