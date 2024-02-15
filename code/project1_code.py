import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from scipy.linalg import svd
import scipy.stats as stat
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


# General summary statistics
        
def continuous_summary_stats():

    for col_name, col_data in zip(continuous_cols, data_drop_missing_obs[:, continuous_idx].T):

        print(col_name)
        print("Mean: " + str(round(np.mean(col_data), 2)))
        print(f"Median: {round(np.median(col_data), 2)}")
        print("Standard deviation:chi "+ str(round(np.std(col_data), 2)))
        print(f"25% percentile: {str(round(np.percentile(col_data, 25), 2))}") 
        print(f"75% percentile: {str(round(np.percentile(col_data, 75), 2))}")

    return None


def categorical_summary_stats():
    for col_name, col_data in zip(cat_cols, data_drop_missing_obs[:, cat_idx].T):
        categories = np.array(list(set(col_data)))
        counts = np.array([np.count_nonzero(col_data==i) for i in categories])
        print(col_name)
        print(categories)
        print(f"Percentages: {np.around(100*counts/sum(counts), 2)}")



# histograms()
# barplots()
# continuous_summary_stats()
# categorical_summary_stats()      

            
# PCA
        
def PCA(save_figs=False):
    X = data_drop_missing_obs[:, continuous_idx]
    y = data_drop_missing_obs[:, -1]
    print(set(y))
    print(X.shape, y.shape)
    N, M = X.shape[0], X.shape[1]
    C = 5

    # Standardising data

    X_s = (X - np.ones((N, 1))*X.mean(axis=0))/X.std(axis=0)

    # Computing SVD
    
    U, S, V = svd(X_s, full_matrices=False)
    rho = (S * S) / (S * S).sum()

    threshold=0.9
    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
    plt.plot([1, len(rho)], [threshold, threshold], "k--")
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()

    if save_figs:
        path = "figures/PCA/cumvar.png"
        plt.savefig(path, dpi=300)

    plt.show()

    # Project the centered data onto principal component space
    Z = X_s @ V.T

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = plt.figure()
    plt.title("Cleveland Heart Disease: PCA")
    # Z = array(Z)
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
    plt.legend([str(i) for i in range(5)])
    plt.xlabel("PC{0}".format(i + 1))
    plt.ylabel("PC{0}".format(j + 1))
    if save_figs:
        path = "figures/PCA/projected.png"
        plt.savefig(path, dpi=300)

    plt.show()


# PCA()