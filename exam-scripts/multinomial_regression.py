import numpy as np

def class_probabilities(x, weights):
    """
    Computes the class probabilities for each class given the input x and the weights.
    - x: data points
    - weights: weights for each class

    Returns:

    - exp_z: the class probabilities for each class
    """
    z = np.dot(x, weights.T)
    exp_z = np.exp(z)
    
    


    return exp_z/np.sum(exp_z, axis=1, keepdims=True)