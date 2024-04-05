import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from scipy.linalg import svd
import scipy.stats as stat
import sys


# Path to data
# PLEASE CHANGE
filepath= ""



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



def is_orthonormal(a):
    """
    Helper function to check correctness of V matrix
    """
    if not np.allclose(np.sum(a**2, axis=0), 1):  # Check if norms are 1
        return False
    if not np.allclose(a.T @ a, np.eye(a.shape[1])):  # Check if dot products are 0
        return False
    return True

# Summary Statistics

def normal_pdf(x, sigma, mu):
    return 297*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
    

# Histograms for continuous data

def histograms(save_figs=False):
    """
    Generates histograms with fitted gaussian density curve.
    """

    for col_name, col_data in zip(continuous_cols, data_drop_missing_obs[:, continuous_idx].T):

        fig, ax = plt.subplots()
        num_bins = 20

        
        n, bins, patches = ax.hist(col_data, num_bins, color='lightsteelblue', edgecolor="black", alpha=0.5)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title(col_name)
        # ax1 = ax.twinx()
        # ax1.set_xlim(min(col_data), max(col_data))

        bin_width = (col_data.max() - col_data.min()) / num_bins
        mean, std, median = np.mean(col_data), np.std(col_data), np.median(col_data)
        
        ax.plot(sorted(col_data), bin_width*normal_pdf(sorted(col_data), std, mean), c="r")
        ax.fill_between(sorted(col_data), bin_width*normal_pdf(sorted(col_data), std, mean), color="lightcoral", alpha=0.2)

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
    """
    Generates bar plots.
    """

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


def continuous_summary_stats():
    """
    General summary statistics 
    """

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


        
def PCA(save_figs=False):
    X = data_drop_missing_obs[:, continuous_idx]
    y = data_drop_missing_obs[:, -1]
    print(set(y))
    print(X.shape, y.shape)
    N, M = X.shape[0], X.shape[1]
    C = 5

    # Standardising data

    X_s = (X - np.ones((N, 1))*X.mean(axis=0))/X.std(axis=0)
    print("X_s:")
    print(X_s)

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


    print("------ Principal Components -------- ")
    print(V.T)

    conv_to_bin = lambda x: 1 if x > 0 else 0

    y_bin = np.array(list(map(conv_to_bin, y)))
    C_bin=2


    print(set(y_bin))

    # Indices of the principal components to be plotted
    i = 3
    j = 0

    # Plot PCA of the data
    f = plt.figure()
    # Z = array(Z)
    colour=['b','r']
    for c in range(C_bin):
        # select indices belonging to class c:
        class_mask = y_bin == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], "o",color=colour[c], alpha=0.5)
    plt.legend(["Healthy", "Unhealthy"])
    plt.title(f"Projection onto PC{j+1} and PC{i+1}")
    plt.xlabel("PC{0}".format(i + 1))
    plt.ylabel("PC{0}".format(j + 1))
    plt.show()

    pcs = [0, 1, 2, 3, 4]

    legendStrs = ["PC" + str(e + 1) for e in pcs]
    bw = 0.2
    r = np.arange(1, M + 1)
    for i in pcs:
        plt.bar(r + i * bw, V.T[:, i], width=bw)
    plt.xticks([0.9,1.9,2.9,3.9,4.9], continuous_cols)
    plt.xlabel("Attributes")
    plt.ylabel("Component coefficients")
    plt.legend(legendStrs)
    plt.grid()
    plt.title("PCA Component Coefficients")
    plt.tight_layout()
    plt.show()

    r = np.arange(1, X.shape[1] + 1)
    plt.bar(r, np.std(X, 0), color=['lightsteelblue', 'lightcoral', 'forestgreen', 'mediumturquoise', 'plum'], alpha=0.5)
    plt.xticks(r, continuous_cols)
    plt.ylabel("Standard deviation")
    plt.xlabel("Attributes")
    plt.title("Attribute standard deviations")
    plt.show()



def effectOfStandardising():
    X = data_drop_missing_obs[:, continuous_idx]
    y = data_drop_missing_obs[:, -1]
    print(set(y))
    print(X.shape, y.shape)
    N, M = X.shape[0], X.shape[1]
    C = 5
    # Subtract the mean from the data
    Y1 = X - np.ones((N, 1)) * X.mean(0)

    # Subtract the mean from the data and divide by the attribute standard
    # deviation to obtain a standardized dataset:
    Y2 = X - np.ones((N, 1)) * X.mean(0)
    Y2 = Y2 * (1 / np.std(Y2, 0))
    # Here were utilizing the broadcasting of a row vector to fit the dimensions
    # of Y2

    # Store the two in a cell, so we can just loop over them:
    Ys = [Y1, Y2]
    titles = ["Zero-mean", "Zero-mean and unit variance"]
    threshold = 0.9
    # Choose two PCs to plot (the projection)
    i = 0
    j = 1

    # Make the plot
    plt.figure(figsize=(10, 15))
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle("Effect of standardization")
    plt.axis('off')
    nrows = 3
    ncols = 2
    for k in range(2):
        # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
        U, S, Vh = svd(Ys[k], full_matrices=False)
        V = Vh.T  # For the direction of V to fit the convention in the course we transpose
        # For visualization purposes, we flip the directionality of the
        # principal directions such that the directions match for Y1 and Y2.
        # if k == 1:
        #     V = -V
        #     U = -U

        # Compute variance explained
        rho = (S * S) / (S * S).sum()

        # Compute the projection onto the principal components
        Z = U * S

        # Plot projection
        plt.subplot(nrows, ncols, 1 + k)
        C = 5
        for c in range(C):
            plt.plot(Z[y == c, i], Z[y == c, j], ".", alpha=0.5)
        plt.xlabel("PC" + str(i + 1))
        plt.ylabel("PC" + str(j + 1))
        plt.title(titles[k] + "\n" + "Projection")
        plt.legend(range(5))
        plt.axis("equal")

        # Plot attribute coefficients in principal component space
        plt.subplot(nrows, ncols, 3 + k)
        for att in range(V.shape[1]):
            plt.arrow(0, 0, V[att, i], V[att, j])
            plt.text(V[att, i], V[att, j], continuous_cols[att])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel("PC" + str(i + 1))
        plt.ylabel("PC" + str(j + 1))
        plt.grid()
        # Add a unit circle
        plt.plot(
            np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01))
        )
        plt.title(titles[k] + "\n" + "Attribute coefficients")
        plt.axis("equal")

        # Plot cumulative variance explained
        plt.subplot(nrows, ncols, 5 + k)
        plt.plot(range(1, len(rho) + 1), rho, "x-")
        plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
        plt.plot([1, len(rho)], [threshold, threshold], "k--")
        plt.title("Variance explained by principal components")
        plt.xlabel("Principal component")
        plt.ylabel("Variance explained")
        plt.legend(["Individual", "Cumulative", "Threshold"])
        plt.grid()
        plt.title(titles[k] + "\n" + "Variance explained")
        
        plt.show()



def genHeatmap():
    """
    Generates heatmap and pairplot using seaborn
    """
    import seaborn as sns

    cleveland_data[continuous_cols].head()

    corr = cleveland_data[continuous_cols].corr()
    print(corr)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, square=True, linewidths=.5, annot=True, vmin=-0.5, vmax=0.5)
    
    plt.show()

    sns.pairplot(cleveland_data[[i for i in continuous_cols]+['num']], hue='num')
    plt.show()