import numpy as np
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

print("\n-----Question 3------\n")

S  = np.array([10.2,6.1,2.8,2.2,1.6])
exp_var = individual_explained_var(S)
cum_var = cum_explained_var(S)

print(f"A: {exp_var[0]} < 0.70")
print(f"B: {cum_var[2]} < 0.90")
print(f"C: {exp_var[4]} < 0.01")

print("\n-----Question 8------\n")

weights = np.array([-46.8, 0.6, -271.1, -31.9, -44.7])
x_low_gears = np.array([6, 120, 3.2, 0, 1])
x_high_gears = np.array([6, 120, 3.2, 0, 10])
z_low = 1257.6 + np.dot(weights, x_low_gears)
z_high = 1257.6 + np.dot(weights, x_high_gears)
print(f"B: low gears: {1/(1+np.exp(-z_low))}, high gears: {1/(1+np.exp(-z_high))}")


x = np.array([6, 120, 3.2, 0, 4])
z = 1257.6 + np.dot(weights, x)
print(f"C: {1/(1+np.exp(-z))} < 0.5")

print("\n-----Question 9------\n")
N = 32
currentWeights = np.array([1/N]*N)
labels = 32*[1]
predictions = 30*[1]+2*[0]
newWeights = updateWeights(currentWeights, predictions, labels)
print(f"New Weights: {newWeights}")

print("\n-----Question 14------\n")

data = np.array([19.4, 30.3, 34.2,38.3,40.1,42.0,50.9,68.6])
labels = np.array([1,0,1,1,1,1,0,0])
k = 2
initial_centroids = np.array([data[0], data[1]])
final_clusters, final_centroids = kmeans(data, k, initial_centroids)
print(f"Final clusters: {final_clusters}")
print(f"Final centroids: {final_centroids}")

print("\n-----Question 15------\n")

Z = np.array([1,1,1,1,0,0,0,1])
Q = np.array([0,1,1,1,0,1,0,1])
print(f"Jaccard Index: {index(Z, Q, 'jaccard')}")

print("\n-----Question 17------\n")

hyp_tang = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

x = np.array([1, 6, 120, 3.2, 0, 4])
w1 = np.array([-4, 1, 0.01, 1, -1, -1])
w2 = np.array([-10, 1, -0.02, 1, 1, 1])

output = 8*hyp_tang(np.dot(w1, x))+9*hyp_tang(np.dot(w2, x))+7
print(f"Output: {output}")

print("\n-----Question 18------\n")

transactions = """1 0 1 0 0 1 
1 0 1 0 0 1 
1 0 1 0 0 1 
1 0 1 0 1 0 
0 1 0 1 1 0 
1 0 0 1 1 0 
0 1 0 1 1 0 
1 0 1 0 1 0"""
transactions = str_to_binary_nparray(transactions)
item_names = "hpL hpH wtL wtH am=0 am=1".split()
confidence = conf(transactions, [3,4], [1])

print("{wtH ,am=0}→{hpH}: ", confidence)