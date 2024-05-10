import numpy as np
import matplotlib.pyplot as plt
"""
Functions for Exam
"""
from bayesian_classifiers import NaiveBayesClassifier
from string_format import str_to_nparray, str_to_binary_nparray
from association_mining import individualSupport, itemsetSupport, conf
from similaritymeasure import *
from EM import multivariate_normal, E_step
from explained_var import individual_explained_var, cum_explained_var
from multinomial_regression import class_probabilities
from K_means import kmeans
from adaboost import getAlpha, majorityVotingClassifier, updateWeights
from AUC_ROC import AUC_no_coords
from draw_dendogram import DrawDendogram
from KNN_density import density, ard
from neural_networks import NN_output
from NeuralNetwork import NeuralNetwork
from comparing_partitions import delta, countingMat, index



print("\n-----Question 2------\n")

S  = np.array([28.4,5.5,1.2,0.5])
exp_var = individual_explained_var(S)
cum_var = cum_explained_var(S)

print(f"A: {exp_var[0]} > 0.95")
print(f"B: {cum_var[1]} > 0.999")
print(f"C: {exp_var[3]} > 0.0005")


print("\n-----Question 7------\n")

D = """0 0.534 1.257 1.671 1.090 1.315 1.484 1.253 1.418
0.534 0 0.727 2.119 1.526 1.689 1.214 0.997 1.056
1.257 0.727 0 2.809 2.220 2.342 1.088 0.965 0.807
1.671 2.119 2.809 0 0.601 0.540 3.135 2.908 3.087
1.090 1.526 2.220 0.601 0 0.331 2.563 2.338 2.500
1.315 1.689 2.342 0.540 0.331 0 2.797 2.567 2.708
1.484 1.214 1.088 3.135 2.563 2.797 0 0.275 0.298 
1.253 0.997 0.965 2.908 2.338 2.567 0.275 0 0.343
1.418 1.056 0.807 3.087 2.500 2.708 0.298 0.343 0"""



D = str_to_nparray(D)

# DrawDendogram(D, method='complete', metric='euclidean')

print("\n-----Question 8------\n")

Z = np.array([1,1,1,2,2,2,3,3,3]) # True labels
Q = np.array([1,1,3,2,2,2,3,3,3]) # Predicted labels
print(f"Rand Index: {index(Z, Q, 'rand')}")

print("\n-----Question 10------\n")
observation = 3
k = 1
print(f"ARD of observation {observation+1} is {ard(D, observation, k)}")



print("\n-----Question 11------\n")

data = np.array([-2.1, -1.7, -1.5, -0.4, 0, 0.6, 0.8, 1, 1.1])
k = 3
initial_centroids = np.array([data[0], data[1], data[2]])
final_clusters, final_centroids = kmeans(data, k, initial_centroids)
print(f"Final clusters: {final_clusters}")
print(f"Final centroids: {final_centroids}")

print("\n-----Question 13------\n")

w_a = np.round(np.linalg.norm(np.array([0.0538, 0.0558, 0.1861, 0.0596])), 4)
w_b = np.round(np.linalg.norm(np.array([0.0089, 0.0931, 0.1093, 0.0417])),4)
w_c = np.round(np.linalg.norm(np.array([0.2811, 0.0445, 0.3379, 0.4626])),4)
w_d = np.round(np.linalg.norm(np.array([0.0167, 0.0698, 0.1354, 0.0403])),4)

print("w_a: ", w_a)
print("w_b: ", w_b)
print("w_c: ", w_c)
print("w_d: ", w_d)




print("\n-----Question 15------\n")
transasctions = """1 1 1 1 0 1 
0 0 0 0 0 0
1 1 0 1 0 0
0 1 1 0 1 0 
1 1 1 1 1 1 
0 0 0 0 0 0
1 1 0 1 0 0
0 1 1 0 1 0 
1 1 1 1 0 1 
0 1 1 0 1 0 
0 0 0 0 0 0 
1 1 0 1 0 0 
0 1 1 0 1 0
0 1 1 0 1 0"""
item_names=np.array("x1 x2 x3 x4 x5 y".split())


transactions = str_to_binary_nparray(transasctions)
y = transactions[:,-1]
print("A:")
print(individualSupport(item_names, transactions))
print(itemsetSupport(transactions, [0,1]))
print(itemsetSupport(transactions, [1,2]))
print(itemsetSupport(transactions, [1,3]))
print(itemsetSupport(transactions, [1,4]))
print(itemsetSupport(transactions, [2,4]))

print("B:")
print("A +")
print(itemsetSupport(transactions, [0, 3]))

print("C:")
print("B +")
print(itemsetSupport(transactions, [1,2,4]))

print("D:")
print("C +")
print(itemsetSupport(transactions, [0,1,3]))

print("\n-----Question 16------\n")
print("A: ", conf(transactions, [0,1,2,3,4],[5]))

print("\n-----Question 17------\n")
X = transactions[:, :2]
y = transactions[:, -1]
NaiveBayesClassifier(transactions, y, np.array([0,1]), np.array([1,1]), 1)

print("\n-----Question 18------\n")
x1 = transactions[:, 0]
for i in range(len(x1)):
    print(f"{x1[i]}: {y[i]}")


print("\n-----Question 19------\n")
r = np.array([1, 1, 1, 1, 0 ,1])
s = np.array([1, 1, 0, 1, 0, 0])
print(f"A: {Jaccard(r, s)} < {SMC(r,s)}")
print(f"B: {Jaccard(r, s)} > {Cosine(r, s)}")
print(f"C: {SMC(r, s)} > {Cosine(r, s)}")
print(f"D: {Cosine(r, s)} = {round(3/15, 4)}")


print("\n-----Question 20------\n")

NN = NeuralNetwork(2, 3, 1)
weights = [np.array([0.5,0.5,-0.5,0.5,-0.5,0.25]), np.array([0.25,0.25,0.25])]
NN.addWeights(weights)
activation_function = lambda x: x if x > 0 else 0
NN.addActivationFunction(activation_function)
x = np.array([1,1])

print(f"Output of Neural Network: {NN.forward(x)}")


print("\n-----Question 25------\n")
N = 25
currentWeights = np.array([1/N]*N)
labels = 25*[1]
predictions = 20*[1]+[0]*5
newWeights = updateWeights(currentWeights, predictions, labels)
print(f"New Weights: {newWeights}")