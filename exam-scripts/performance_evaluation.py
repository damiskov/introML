import numpy as np
from probability_distributions import *

def mcnemars(Z, A, B, alpha=0.05):
    """
    Performs the McNemar's test for two classifiers.

    Parameters
    ----------
    Z: numpy.ndarray
        Ground truth partition.
    A: numpy.ndarray
        A partition.
    B: numpy.ndarray
        Another partition.

    Returns
    -------
    theta_hat: float
        Estimated performance difference A-B
    (theta_L, theta_U): tuple
        Confidence interval.
    p_value: float
        The p-value of the test.
    """

    N = len(Z)
    n = np.zeros((2, 2))
    for i in range(N):
        if A[i] == Z[i] and B[i] == Z[i]:
            n[0, 0] += 1
        elif A[i] == Z[i] and B[i] != Z[i]:
            n[0, 1] += 1
        elif A[i] != Z[i] and B[i] == Z[i]:
            n[1, 0] += 1
        elif A[i] != Z[i] and B[i] != Z[i]:
            n[1, 1] += 1
        
    theta_hat = (n[0, 1] + n[1, 0]) / N
    E_theta = (n[0, 1] - n[1, 0]) / N
    Q = (N**2(N+1)*(E_theta+1)*(E_theta-1))/(N*(n[0, 1] + n[1, 0])-(n[0, 1] - n[1, 0])**2)
    f = (E_theta+1)*(Q-1)/2
    g = (1-E_theta)*(Q-1)/2

    theta_L = 2*Beta_cdf(alpha/2, f, g) -1
    theta_U = 2*Beta_cdf(1-alpha/2, f, g) -1

    p_value = 2*binomial_cdf(min(n[0, 1], n[1, 0]), n[0, 1] + n[1, 0], 0.5)

    return theta_hat, (theta_L, theta_U), p_value

