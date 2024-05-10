import numpy as np

def cum_explained_var(S):
    """
    Computes the explained variance for first 1, 1-2, 1-3... principal components given the singular values S.
    """
    cum_var = np.cumsum(S**2)/np.sum(S**2)
    return np.round(cum_var, 3)

def individual_explained_var(S):
    """
    Computes the explained variance for each principal component given the singular values S.
    """
    ind_var = S**2/np.sum(S**2)
    return np.round(ind_var, 3)

if __name__ == "__main__":
    
    S = np.array([149,118,53,42,3])
    print("Individual explained variance: ", individual_explained_var(S))
    print("Cumulative explained variance: ", cum_explained_var(S))