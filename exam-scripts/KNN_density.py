"""
Script containing functions that calculate the KNN density of a point in a dataset or it's average relative density given a distance matrix

"""
import numpy as np
from string_format import str_to_nparray

def density(D, i, k):
    """
    Computes the KNN density of a given point in the dataset given the distance matrix D.
    """
    # make distance vector, with 0 distance (point->same point) removed
    distances_i = np.delete(D[:, i], i)
    # sort the distances
    distances_i = np.sort(distances_i)
    # return the density of the point
    return 1/(sum(distances_i[:k])/(k))



    
def ard(D, i, k):
    """
    Computes the average relative density of a given point in the dataset given the distance matrix D.

    Parameters:
        - D: numpy array. Distance matrix of the dataset.
        - i: int. Index of the point in the dataset. (0-indexed)
        - k: int. Number of nearest neighbors to consider.
    
    """
    # Find indices of k nearest neighbors, ignoring the point itself
    knn_indices = np.argsort(D[:, i])[1:k+1]

    # Compute the density of the point
    point_density = density(D, i, k)

    # Compute the density of the k nearest neighbors
    knn_densities = np.array([density(D, j, k) for j in knn_indices])

    # Compute the average relative density  
    return np.round(point_density/np.mean(knn_densities), 4)

    


if __name__ == "__main__":

    table = """
    0.00 4.84 0.50 4.11 1.07 4.10 4.71 4.70 4.93 
    4.84 0.00 4.40 5.96 4.12 2.01 5.36 3.59 3.02 
    0.50 4.40 0.00 4.07 0.72 3.75 4.66 4.48 4.64 
    4.11 5.96 4.07 0.00 4.48 4.69 2.44 3.68 4.15 
    1.07 4.12 0.72 4.48 0.00 3.54 4.96 4.62 4.71 
    4.10 2.01 3.75 4.69 3.54 0.00 3.72 2.23 1.95 
    4.71 5.36 4.66 2.44 4.96 3.72 0.00 2.03 2.73 
    4.70 3.59 4.48 3.68 4.62 2.23 2.03 0.00 0.73 
    4.93 3.02 4.64 4.15 4.71 1.95 2.73 0.73 0.00"""
    table = str_to_nparray(table)


    # REMEMBER: 0-indexed
    print("Average Relative Density: ", ard(table,8, 2))





    


