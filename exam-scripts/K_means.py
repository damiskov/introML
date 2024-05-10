import numpy as np
from scipy.spatial import distance

def kmeans(data, k, initial_centroids, num_iter=100):
    """
    does k-means clustering on the data for a given number of iterations.

    Parameters:
        data: a list of lists of floats
        k: an integer
        num_iter: an integer
       initial_centroids: a list of floats
    
    Returns:
        final_centroids: a list of floats
    """
    #initialize the clusters
    clusters = [[] for _ in range(k)]
    final_clusters = [[] for _ in range(k)]
    #iterate over the number of iterations
    centroids = initial_centroids
    for i in range(num_iter):

        #assign each data point to the closest centroid
        for j in range(len(data)):
            distances = np.array([np.abs(data[j]-centroids[l]) for l in range(k)])
            clusters[np.argmin(distances)].append(data[j])
        
        #update the centroids
        
        for j in range(k):
            if len(clusters[j]) > 0:
                centroids[j] = np.mean(clusters[j])
        #reset the clusters
        if i != num_iter-1:
            final_clusters = clusters
        clusters = [[] for _ in range(k)]
    
    final_centroids = centroids

    return final_clusters, np.round(final_centroids, 3)


# def kmeans(X,num_clusters, num_iter=100):
#     """
#     Given 1-d dataset X, returns the final centroids of the clusters formed by K-means clustering.
    
#     Parameters:
#     - X: a list of floats
#     - num_clusters: an integer
#     Returns:
#     - final_clusters: a list of lists of floats
#     - final_centroids: a list of floats
#     """
#     #initialize the clusters
#     clusters = [[] for _ in range(num_clusters)]
#     #initialize the centroids randomly in data
#     initial_centroids = np.random.choice(X, num_clusters)
#     #iterate over the number of iterations
#     centroids = initial_centroids
#     for _ in range(100):

#         #assign each data point to the closest centroid
#         for j in range(len(X)):
#             distances = np.array([np.abs(X[j]-centroids[l]) for l in range(num_clusters)])
#             clusters[np.argmin(distances)].append(X[j])
        
#         #update the centroids
        
#         for j in range(num_clusters):
#             if len(clusters[j]) > 0:
#                 centroids[j] = np.mean(clusters[j])
#         #reset the clusters
#         clusters = [[] for _ in range(2)]
    
#     final_centroids = centroids
#     final_clusters = clusters

#     return final_clusters, final_centroids