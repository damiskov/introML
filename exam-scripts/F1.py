import numpy as np

"""
Simple function to calculate F1 score. 2*precision*recall/(precision+recall)
"""

def f1_score(tp, fp, fn):
    """
    Calculate the F1 score given the number of true positives, false positives, and false negatives.
    """
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return 2*precision*recall/(precision+recall)

if __name__ == "__main__":
    
    mat = np.array([[34, 11],
                    [7, 39]])
    
    tp = mat[0, 0]
    fp = mat[0, 1]
    fn = mat[1, 0]

    print(f1_score(tp, fp, fn))