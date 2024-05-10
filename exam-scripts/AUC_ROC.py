import numpy as np
import matplotlib.pyplot as plt

def get_ROC_coord(x, y):
    """
    Compute a coordinate of ROC curve based on a given classifier's output.

    Parameters
    ----------
    y: numpy.ndarray
        The true labels of the data.
    x: numpy.ndarray
        The predicted labels of the data.
    
    Returns
    -------
    FPR, TPR: float
        The false positive rate and true positive rate of the classifier. (Coordinates of ROC curve)
    """
    TP = np.sum((x == 1) & (y == 1))
    FP = np.sum((x == 1) & (y == 0))
    TN = np.sum((x == 0) & (y == 0))
    FN = np.sum((x == 0) & (y == 1))


    # Compute the false positive rate and true positive rate
    FPR = FP / (FP+TN)
    TPR = TP / (TP+FN)

    return round(FPR, 3), round(TPR, 3)


def calc_AUC(labels, x_coords):
    """
    Compute the area under the ROC curve.

    Parameters
    ----------
    labels: numpy.ndarray
        The true labels of the data.
    x_coords: numpy.ndarray
        Continuous data associated with true labels.

    Returns
    -------
    float
        The area under the ROC curve.
    """
    ROC = np.empty((len(x_coords), 2))
    thresholds = np.linspace(min(x_coords), max(x_coords)+0.2*max(x_coords), len(x_coords))

    for i, threshold in enumerate(thresholds):
        x = np.array([1 if i > threshold else 0 for i in x_coords])
        ROC_x, ROC_y = get_ROC_coord(x, labels)
        ROC[i, 0] = ROC_x
        ROC[i, 1] = ROC_y

    # Compute the area under the ROC curve
    AUC = 0
    for i in range(1, len(ROC)):
        AUC += (ROC[i, 0] - ROC[i-1, 0]) * (ROC[i, 1] + ROC[i-1, 1]) / 2

    return 1+round(AUC, 3)

def regen_ROC(labels, x_coords, n=50):
    """
    Recreates ROC curve based on data points.

    Parameters
    ----------
    labels: numpy.ndarray
        The true labels of the data.
    x_coords: numpy.ndarray
        Continuous data associated with true labels.

    Returns
    -------
    Graph
        ROC curve.
    """

    ROC = np.empty((n, 2))
    vlines = []
    thresholds = np.linspace(min(x_coords), max(x_coords)+0.2*max(x_coords), n)

    for i, threshold in enumerate(thresholds):
        x = np.array([1 if i > threshold else 0 for i in x_coords])
        ROC_x, ROC_y = get_ROC_coord(x, labels)
        ROC[i, 0] = ROC_x
        ROC[i, 1] = ROC_y

    
    # plotting

    plt.plot(ROC[:,0], ROC[:,1])
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('TPR')
    plt.grid()
    plt.tight_layout()
    plt.show()



def plot_tpr_fpr(labels, coords):
    """
    Simple function for plotting TPR and FPR.

    Parameters
    ----------
    labels: numpy.ndarray
        The true labels of the data.
    coords: numpy.ndarray

    Returns
    -------
    Graph
        TPR and FPR.
    """

    thresholds = np.linspace(min(coords), max(coords)+0.2*max(coords), len(coords))
    FPR = np.empty(len(thresholds))
    TPR = np.empty(len(thresholds))
    ROC = np.empty((len(thresholds), 2))

    for i, threshold in enumerate(thresholds):
        x = np.array([1 if i > threshold else 0 for i in coords])
        FPR[i], TPR[i] = get_ROC_coord(x, labels)
        ROC[i, :] = np.array([FPR[i], TPR[i]])

    plt.plot(FPR, TPR)
    plt.title('TPR and FPR')
    plt.xlabel('FPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('TPR')
    plt.grid()
    plt.tight_layout()
    plt.show()



def AUC_no_coords(labels,predictions):
    """
    Compute the area under the ROC curve.

    Parameters
    ----------
    labels: numpy.ndarray
        The true labels of the data.
    predictions: numpy.ndarray
        The predicted labels of the data.

    Returns
    -------
    float
        The area under the ROC curve.
    """

    TPR = np.empty(len(labels))
    FPR = np.empty(len(labels))

    for i in range(len(labels)):
        x = np.array([1 if j > predictions[i] else 0 for j in predictions])
        FPR[i], TPR[i] = get_ROC_coord(x, labels)
    
    AUC = 0

    for i in range(1, len(FPR)):
        AUC += (FPR[i] - FPR[i-1]) * (TPR[i] + TPR[i-1]) / 2


    return round(AUC, 3)


if __name__ == "__main__":
    # Example usage
    l2 = np
    x_coords = np.array([0.1,0.11,0.12,0.13,0.14,0.18,0.2,0.7,0.75,0.9,0.95])

    if len(l1) != len(x_coords):
        raise ValueError("The length of labels and x_coords must be equal.")
    print("AUC:", calc_AUC(l1, x_coords))
    regen_ROC(l1, x_coords)
