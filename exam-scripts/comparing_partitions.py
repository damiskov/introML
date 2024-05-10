import numpy as np

def delta(i, j):
    return 1 if i == j else 0

def countingMat(Z, Q):
    """
    Given two partitions Z and Q, return the counting matrix.

    Parameters
    ----------
    Z: numpy.ndarray
        A partition.
    Q: numpy.ndarray 
        Another partition.
    
    Returns
    -------
    numpy.ndarray
        The counting matrix.
    """

    K = len(set(Z))
    M = len(set(Q))
    N = len(Z)
    C = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            C[k, m] = sum([Z[i] == k and Q[i] == m for i in range(N)])
    return C

def index(Z, Q, method):
    """
    Given two partitions Z and Q, return the Rand Index (SMC).

    Parameters
    ----------
    Z: numpy.ndarray
        A partition.
    Q: numpy.ndarray 
        Another partition.
    method: str
        - 'rand': Rand Index
        - 'jaccard': Jaccard Index
    
    Returns
    -------
    float
        The Rand Index.
    """

    S, D = 0, 0
    N = len(Z)
    for i in range(N-1):
        for j in range(i+1, N):
            S += delta(Z[i], Z[j])*delta(Q[i], Q[j])
            D += (1 - delta(Z[i], Z[j]))*(1 - delta(Q[i], Q[j]))

    
    if method == 'rand':
        return round((S+D)/(0.5*N*(N-1)), 4)
    elif method == 'jaccard':
        return round(S/(0.5*N*(N-1) - D), 4)
    else:
        raise ValueError("Invalid method.")

def normalisedMutualinformation(Z, Q):
    """
    Given two partitions Z and Q, return the Normalised Mutual Information.
    
    Basically just returns the normalised counting matrix.

    Parameters
    ----------
    Z: numpy.ndarray
        A partition.
    Q: numpy.ndarray 
        Another partition.
    
    Returns
    -------
    matrix
        The Normalised Mutual Information. 
    """

    N = len(Z)
    C = countingMat(Z, Q)
    
    return C/N