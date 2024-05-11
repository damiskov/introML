import numpy as np
from string_format import str_to_binary_nparray

def bayesClassifier(table, labels, X, y):

    """
    Given a table of binary values and labels, and a new observation X, return the predicted label for X using the Naive Bayes classifier.
    """
    
    return

def NaiveBayesClassifier(X, y, f_index, f_labels, y_test):
    """

    Performs the Naive Bayes Classifier on a given table of binary values and labels.
    
    Parametersa
    ----------
    X: nparray
        The table of binary values. (transactions)
    y: nparray
        The associated classes of each observation.
    feature_idx: nparray
        The indices of the features to consider. (f_1, f_2, ..., 0-indexed!)
    feature_vals: nparray
        The labels of the features to consider. (f_1=0, f_2=1, ... etc)
    class_prediction: int, the label of the test observation.

    Returns:
    -------
    float
        The probability that an observation, given features (feature_idx) with 
        values (feature_vals), belongs to a given class (class_prediction).
    """
    


    # Calculate the prior probabilities
    p_y = np.sum(y==y_test)/len(y)
    p_not_y = np.sum(y!=y_test)/len(y)


    # Numerator

    num = np.zeros(len(f_index))
    for j, fidx in enumerate(f_index):
        for i in range(len(X)):
            if X[i][fidx] == f_labels[j] and y[i] == y_test:
                num[j] += 1
        num[j] /= np.sum(y==y_test)

    num = np.prod(num)*p_y

    # Denominator

    den = np.zeros(len(np.unique(y)))
    for i, label in enumerate(np.unique(y)):
        temp = np.zeros(len(f_index))
        for j, fidx in enumerate(f_index):
            temp_f = 0
            for k in range(len(X)):
                if X[k][fidx] == f_labels[j] and y[k] == label:
                    temp_f += 1
            temp_f /= np.sum(y==label)
            temp[j] = temp_f
        den[i] = np.prod(temp)*np.sum(y==label)/len(y)

    den = np.sum(den)

    NB_prob = num/den


    print(f"Probability of y={y_test}")
    print(f"Given:")
    for i in range(len(f_index)):
        print(f"\tf_{f_index[i]+1}={f_labels[i]}")
    
    print(f"p: {round(NB_prob, 3)}")
    return NB_prob



if __name__=="__main__":

    print(" Example usage (Question 20, Fall 2017):\n")
    transactions = """1 0 1 0 1 0 1 0
    1 0 1 0 1 0 1 0
    1 0 1 0 1 0 1 0
    1 0 1 0 1 0 0 1
    1 0 1 0 0 1 0 1
    1 0 0 1 0 1 1 0
    0 1 1 0 0 1 0 1
    0 1 1 0 1 0 0 1
    0 1 0 1 1 0 1 0
    0 1 0 1 0 1 1 0
    """
    """
    The above must be converted to a numpy array of integers.
    """
    transactions = str_to_binary_nparray(transactions)
    print("\n-----Question 20------\n")
    labels = np.array([2,1,1,2,3,1,3,3,1,3])
    feature_idx = np.array([1, 2])
    feature_vals = np.array([1, 1])
    class_prediction = 3
    NaiveBayesClassifier(transactions, labels, feature_idx, feature_vals, class_prediction)
