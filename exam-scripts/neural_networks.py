import numpy

def NN_output(weights, inputs, activation_function):
    """
    Compute the output of a neural network with one hidden layer.
    
    Parameters
    ----------
    weights: list of numpy.ndarray
        The weights of the neural network.
    inputs: numpy.ndarray
        The input data.
    activation_function: function
        The activation function of the hidden layer.
    
    Returns
    -------
    numpy.ndarray
        The output of the neural network.
    """
    # Compute the output of the hidden layer
    hidden_output = activation_function(numpy.dot(inputs, weights[0]))
    # Compute the output of the neural network
    output = numpy.dot(hidden_output, weights[1])

    return activation_function(output)