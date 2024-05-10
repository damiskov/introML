"""
Given a distance matrix, this function will draw a dendrogram.

"""
import numpy as np
from matplotlib.pyplot import figure, show
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from string_format import str_to_nparray



def DrawDendogram(X, method='complete', metric='euclidean'):
    """
    Given a distance matrix, this function will draw a dendrogram.

    Parameters
    ----------
    X : ndarray
        A distance matrix.
    method : str
        The linkage method to use. Default is 'complete'.
        - 'single': Nearest Point Algorithm
        - 'complete': Farthest Point Algorithm
        - 'average': UPGMA
        - 'weighted': WPGMA
        - 'centroid': UPGMC
        - 'median': WPGMC
        - 'ward': Incremental Sum of Squares
    metric : str
        The distance metric to use. Default is 'euclidean'.

    Returns
    -------
    None
    """
    z = linkage(squareform(X), method=method, metric=metric, optimal_ordering=True)
    figure(2, figsize=(10, 4))
    dendrogram(z, count_sort='descendent', labels=list(range(1, len(X[0]) + 1)))
    show()


if __name__ == "__main__":

    distances = """
    0.00 4.84 0.50 4.11 1.07 4.10 4.71 4.70 4.93 
    4.84 0.00 4.40 5.96 4.12 2.01 5.36 3.59 3.02 
    0.50 4.40 0.00 4.07 0.72 3.75 4.66 4.48 4.64 
    4.11 5.96 4.07 0.00 4.48 4.69 2.44 3.68 4.15 
    1.07 4.12 0.72 4.48 0.00 3.54 4.96 4.62 4.71 
    4.10 2.01 3.75 4.69 3.54 0.00 3.72 2.23 1.95 
    4.71 5.36 4.66 2.44 4.96 3.72 0.00 2.03 2.73 
    4.70 3.59 4.48 3.68 4.62 2.23 2.03 0.00 0.73 
    4.93 3.02 4.64 4.15 4.71 1.95 2.73 0.73 0.00"""
    
    X = str_to_nparray(distances)

    draw_dendrogram(X, method='single', metric='euclidean')