import numpy as np

def makeLabels(num_0, num_1):
    """
    Create an array of labels with num_0 0's and num_1 1's
    """
    labels = np.zeros(num_0 + num_1)
    labels[num_0:] = 1
    return labels

def impurity(N, labels, method):
    """
    Calculate the impurity of a node
    """
    # Calculate the proportion of 0's and 1's in the node
    p_0 = sum(labels==0)/N
    p_1 = sum(labels==1)/N
    # Calculate the impurity of the node
    if method == "gini":
        impurity = 1 - p_0**2 - p_1**2
    elif method == "entropy":
        impurity = -p_0*np.log2(p_0) - p_1*np.log2(p_1)
    elif method == "classerror":
        impurity = 1 - max(p_0, p_1)
    else:
        raise ValueError("Method must be 'gini', 'entropy', or 'classerror'")
    return impurity

def purityGain(root, left, right, method):
    """
    Calculate the purity gain of a split
    """
    # Calculate the impurity of the root node
    impurity_root = impurity(N_root, labels_root, method)
    # Calculate the impurity of the left child
    impurity_l = impurity(N_l, labels_l, method)
    # Calculate the impurity of the right child
    impurity_r = impurity(N_r, labels_r, method)
    # Calculate the purity gain
    purity_gain = impurity_root - (N_l/N_root)*impurity_l - (N_r/N_root)*impurity_r
    return purity_gain


if __name__=="__main__":
    
    root = makeLabels(8, 3)
    left = makeLabels(6, 1)
    right = makeLabels(2, 2)


    # Calculate purity gain
    print(purityGain(N_r, labels_r, N_l, labels_l, N_r, labels_r, "gini"))
