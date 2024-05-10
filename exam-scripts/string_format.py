import numpy as np

def str_to_nparray(input_string):
    """
    Converts a string of numbers into a numpy array.

    Parameters
    ----------
    input_string: str
        A multiline string of numbers.

    Returns
    -------
    numpy.ndarray
        A numpy array of the numbers.
    """
    rows = input_string.strip().split('\n')
    for i in range(len(rows)):
        rows[i] = rows[i].strip().split(' ')

    return np.array(rows, dtype=np.float64) 


def str_to_binary_nparray(input_string):
    """
    Converts a string of binary numbers into a numpy array.

    Parameters
    ----------
    input_string: str
        A multiline string of binary numbers.

    Returns
    -------
    numpy.ndarray
        A numpy array of the binary numbers.
    """
    rows = input_string.strip().split('\n')
    for i in range(len(rows)):
        rows[i] = rows[i].strip().split(' ')

    return np.array(rows, dtype=np.int32)