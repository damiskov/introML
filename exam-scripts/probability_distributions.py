import numpy as np

def multivariate_normal(mu, sigma,x, d):
    """
    Calculates the probability density function of a multivariate normal distribution.

    Parameters:
        - mu: numpy array of shape (d,). The mean of the distribution.
        - sigma: numpy array of shape (d, d). The covariance matrix of the distribution.
        - x: numpy array of shape (d,). The point at which to evaluate the density function.
        

    Returns:
        - pdf: float. The value of the probability density function at point x.
    """
    pdf = 1/((2*np.pi)**(d/2) * np.linalg.det(sigma)**0.5) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu))
    return pdf

def binomial_pmf(m, N, theta):
    """
    Calculates the probability mass function of a binomial distribution.

    Parameters:
        - m: int. The number of successes.
        - N: int. The number of trials.
        - theta: float. The probability of success.
    

    Returns:
        - pmf: float. The value of the probability mass function at point k.
    """
    return np.math.comb(N, m) * theta**m * (1-theta)**(N-m)

def binomial_cdf(m, N, theta):
    """
    Calculates the cumulative distribution function of a binomial distribution.

    Parameters:
        - n: int. The number of trials.
        - p: float. The probability of success.
        - k: int. The number of successes.

    Returns:
        - cdf: float. The value of the cumulative distribution function at point k.
    """
    return np.sum([binomial_pmf(i, N, theta) for i in range(m+1)])

def Beta_pdf(theta, alpha, beta):
    """
    Calculates the probability density function of a beta distribution.

    Parameters:
        - theta: float. The point at which to evaluate the density function.
        - alpha: float. The alpha parameter of the distribution.
        - beta: float. The beta parameter of the distribution.

    Returns:
        - pdf: float. The value of the probability density function at point theta.
    """
    numerator = np.math.gamma(alpha + beta) * theta**(alpha-1) * (1-theta)**(beta-1)
    denominator = np.math.gamma(alpha) * np.math.gamma(beta)

    return numerator / denominator

def Beta_cdf(theta, alpha, beta):
    """
    Calculates the cumulative distribution function of a beta distribution.

    Parameters:
        - theta: float. The point at which to evaluate the density function.
        - alpha: float. The alpha parameter of the distribution.
        - beta: float. The beta parameter of the distribution.

    Returns:
        - cdf: float. The value of the cumulative distribution function at point theta.
    """
    return np.sum([Beta_pdf(theta, alpha, beta) for i in np.linspace(0, theta, 1000)])