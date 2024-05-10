import numpy as np
from string_format import str_to_nparray


"""
Given table of distances, E.g., 
    o1 o2 o3 o4 o5 o6 o7 o8 o9
o1 0.00 4.84 0.50 4.11 1.07 4.10 4.71 4.70 4.93 
o2 4.84 0.00 4.40 5.96 4.12 2.01 5.36 3.59 3.02 
o3 0.50 4.40 0.00 4.07 0.72 3.75 4.66 4.48 4.64 
o4 4.11 5.96 4.07 0.00 4.48 4.69 2.44 3.68 4.15 
o5 1.07 4.12 0.72 4.48 0.00 3.54 4.96 4.62 4.71 
o6 4.10 2.01 3.75 4.69 3.54 0.00 3.72 2.23 1.95 
o7 4.71 5.36 4.66 2.44 4.96 3.72 0.00 2.03 2.73 
o8 4.70 3.59 4.48 3.68 4.62 2.23 2.03 0.00 0.73 
o9 4.93 3.02 4.64 4.15 4.71 1.95 2.73 0.73 0.00

and K, return the error rate of hold-one-out cross-validation for k-NN with k=K.
"""

def knn_CV(table,labels, K):
    error = 0
    n = len(table)
    for obs in range(n):
        distances = table[obs]
        indices = np.argsort(distances)
        indices = indices[1:K+1]
        votes = labels[indices]
        counts = np.bincount(votes)
        prediction = np.argmax(counts)
        error += prediction != labels[obs]
    return error/n


if __name__=="__main__":
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

    labels = np.array([0,0,0,0,0,1,1,1,1])

    table = str_to_nparray(table)
    K = 3


    o2 =3.04
    o3=1.5
    do2o3 = 4.4

    print((o2**2 + o3**2 - do2o3**2)/(2*o2*o3))

