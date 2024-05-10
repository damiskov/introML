import numpy as np

def bayesClassifier(table, labels, X, y):

    """
    Given a table of binary values and labels, and a new observation X, return the predicted label for X using the Naive Bayes classifier.
    """
    
    return

def NaiveBayesClassifier(X, y, f_index, f_labels, y_test):
    """
    Given a table of binary values and labels, and a new observation X, 
    return the probability of an observation with X_test and y_test 
    using Naive Bayes classifier.

    p(y=y_test | f_1=f_labels[0], ..., f_n=f_labels[n-1])  etc.
    
    Parameters
    ----------
    X: nparray
        The table of binary values. (transactions)
    y: nparray
        The labels of the data.
    f_index: nparray
        The indices of the features to consider. (f_1, f_2, ..., 0-indexed)
    f_labels: nparray
        The labels of the features to consider. (f_1=0, f_2=1, ... etc)
    y_test: int, the label of the test observation.

    (Prints)
    -------
    float
        The probability of the test observation with the given attribute values.

    """
    


    # Calculate the prior probabilities
    p_y = np.sum(y==y_test)/len(y)
    p_not_y = np.sum(y!=y_test)/len(y)

    # Calculate the likelihoods

    p_x_given_y = []
    p_x_given_not_y = []

    for i in range(len(f_index)):
        p_x_given_y.append(np.sum(X[y==y_test][:,f_index[i]] == f_labels[i])/np.sum(y==y_test))
        p_x_given_not_y.append(np.sum(X[y!=y_test][:,f_index[i]] == f_labels[i])/np.sum(y!=y_test))
    



    NB_prob = (p_y * np.prod(p_x_given_y) )/(np.prod(p_x_given_not_y)*p_not_y + p_y * np.prod(p_x_given_y))

    print(f"Probability of y={y_test}")
    print(f"Given:")
    for i in range(len(f_index)):
        print(f"\tf_{f_index[i]+1}={f_labels[i]}")
    
    print(f"p: {round(NB_prob, 3)}")