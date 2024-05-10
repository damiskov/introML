import numpy as np
"""
Script for various functions related to AdaBoost.
"""


def getAlpha(errors):
    """
    Given a list of errors for t=1,2,...,T iterations, returns the list of alphas for each iteration.

    Parameters:
        - errors: list of floats. The errors for t=1,2,...,T iterations.
    
    Returns:
        - alphas: list of floats. The alphas (importance weights) for each iteration.
    """

    alphas = []
    for t in range(len(errors)):
        alphas.append(0.5 * np.log((1 - errors[t]) / errors[t]))
    return alphas

def majorityVotingClassifier(predictions, alphas):
    """
    Given a list of predictions for t=1,2,...,T iterations, a list of labels, and a list of alphas, returns the majority voting classifier for a specific point.

    Only works with binary classification at the moment.

    Parameters:
        - predictions: list of lists of floats. The predicted value for each t=1,2,...,T iterations.
        - alphas: list of importance weights
    
    Returns:
        - MVC: Majority voting class based on adaBoost
    """

    # y = 0 (first class)
    deltas0 = [1 if p == 0 else 0 for p in predictions]
    # y = 1 (second class)
    deltas1 = [1 if p == 1 else 0 for p in predictions]

    # Sums
    S0 = np.sum([deltas0[i] * alphas[i] for i in range(len(predictions))])
    S1 = np.sum([deltas1[i] * alphas[i] for i in range(len(predictions))])

    if S0 > S1:
        return 0
    else:
        return 1
    


def updateWeights(currentWeights, predictions, labels):
    """
    Updates weights for the next iteration of AdaBoost.

    Parameters:
        - currentWeights: list of floats. The current weights for the training data.
        - predictions: list of floats. The predictions for the current iteration.
        - labels: list of floats. The true labels for the training data.
    
    Returns:
        - updatedWeights: list of floats. The updated weights for the training data.
    """
    N = len(currentWeights)
    updatedWeights = []
    # Calculate weighted error
    err_t = 0
    for i in range(N):
        if predictions[i] != labels[i]:
            err_t += currentWeights[i]
     
    alpha_t = 0.5 * np.log((1 - err_t) / err_t)

    for i in range(N):
        if predictions[i] == labels[i]:
            updatedWeights.append(currentWeights[i] * np.exp(-alpha_t))
        else:
            updatedWeights.append(currentWeights[i] * np.exp(alpha_t))

    # Normalize
    updatedWeights = updatedWeights / np.sum(updatedWeights)
    return np.round(updatedWeights, 5)   

if __name__=="__main__":

    predictions1 = [0,1,0,0]
    predictions2 = [0,1,1,1]
    errors = [0.417, 0.243, 0.307, 0.53]

    alphas = getAlpha(errors)

    print('Alphas:', alphas)

    print('Majority voting classifier (y1):', majorityVotingClassifier(predictions1, alphas))
    print('Majority voting classifier (y2):', majorityVotingClassifier(predictions2, alphas))



    