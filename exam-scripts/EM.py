import numpy as np

def multivariate_normal(x, mu, sigma):
    """
    Calculates the probability density function of a multivariate normal distribution.

    Parameters:
        - x: numpy array of shape (d,). The point at which to evaluate the density function.
        - mu: numpy array of shape (d,). The mean of the distribution.
        - sigma: numpy array of shape (d, d). The covariance matrix of the distribution.

    Returns:
        - pdf: float. The value of the probability density function at point x.
    """
    d = x.shape[0]
    pdf = 1/((2*np.pi)**(d/2) * np.linalg.det(sigma)**0.5) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu))
    return pdf

def E_step(X, mu, sigma, pi):
    """
    Performs the E-step of the EM algorithm.

    Parameters:
        - X: numpy array of shape (n, d). The data.
        - mu: numpy array of shape (k, d). The means of the k Gaussian distributions.
        - sigma: numpy array of shape (k, d, d). The covariance matrices of the k Gaussian distributions.
        - pi: numpy array of shape (k,). The mixing coefficients.

    Returns:
        - gamma: numpy array of shape (n, k). The responsibilities.
    """
    n, d = X.shape
    k = mu.shape[0]
    gamma = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            gamma[i, j] = pi[j] * multivariate_normal(X[i], mu[j], sigma[j])
        gamma[i] /= np.sum(gamma[i])
    return gamma