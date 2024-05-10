import numpy as np


def logistic_sigmod(x):
    """
    Compute the logistic sigmoid function.
    """
    return 1/(1 + np.exp(-x))

def simple_lr(x1, x2, x3, x4, x5, c1, c2, c3, c4, c5):
    return 1.41 + 0.76*x1 + 1.76*x2 - 0.32*x3 - 0.96*x4 + 6.64*x5 - 5.13*c1 - 2.06*c2 + 96.73*c3 + 1.03*c4 - 2.74*c5

if __name__=="__main__":
    x6 = simple_lr(-0.06,  0.28, 0.43, -0.30, -0.36, 0 ,0, 0, 0, 1)
    print(logistic_sigmod(x6))
    