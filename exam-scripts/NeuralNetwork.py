import numpy as np

class NeuralNetwork:

    """
    Very simple Neural network with 1 hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = lambda x: x
        self.biases = [np.zeros(hidden_size), np.zeros(output_size)]
    
    def addWeights(self, weights):
        if len(weights) != 2:
            raise ValueError("Must have 2 sets arrays of weights.")
        self.weights = weights

    def addBiases(self, biases):
        if len(biases) != 2:
            raise ValueError("Must have 2 arrays of biases.")
        self.biases = biases

    def addActivationFunction(self, activation_function):
        self.activation_function = activation_function
    
    def forward(self, inputs):
        """
        Compute the output of a neural network with one hidden layer.
        
        Parameters
        ----------
        inputs: numpy.ndarray
            The input data.
        
        Returns
        -------
        numpy.ndarray
            The output of the neural network.
        """
        # 
        out1 = np.zeros(self.hidden_size)
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                out1[j] += inputs[i] * self.weights[0][i+j]
            
        out1 = np.array([self.activation_function(out1[i])+self.biases[0][i] for i in range(self.hidden_size)])
        out2 = np.zeros(self.output_size)
        for j in range(self.output_size):
            for i in range(self.hidden_size):
                out2[j] += out1[i] * self.weights[1][i+j]
        
        out2 = np.array([self.activation_function(out2[i]) + self.biases[1][i] for i in range(self.output_size)])
   
        return out2

    
