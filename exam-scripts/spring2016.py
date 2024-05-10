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
# Question 2

S  = np.array([149,118,53,42,3])
exp_var = individual_explained_var(S)
cum_var = cum_explained_var(S)
print("Individual explained variance: ", exp_var)
print("Cumulative explained variance: ", cum_var)

print(f"A: {sum(exp_var[-3:])} < 0.1")
print(f"B: {exp_var[0]} > 0.6")
print(f"C: {sum(exp_var[-2:])} < 0.04")
print(f"D: {sum(exp_var[:2])} > 0.85")



# Question 10

print("\n-----Question 10------\n")

s = """0 1 1 0 1 
0 0 1 0 0
1 0 0 0 1
1 0 0 1 1
1 0 0 1 0 
1 1 0 1 1 
1 0 1 0 0
1 0 1 1 1
0 1 1 1 1
1 0 1 1 0
0 1 1 0 0"""
labels = np.array([0,0,0,0,0,0,1,1,1,1,1])
tab4 = str_to_binary_nparray(s)

NaiveBayesClassifier(tab4, labels, np.array([0,1,2]), np.array([0,1,1]), 1)

# Question 11

print("\n-----Question 11------\n")

transactions = tab4
item_names = "f1 f2 f3 f4 f5".split()
print("A:")
print(individualSupport(item_names, transactions))
print("B:")
print(f"A + {itemsetSupport(transactions, [0,3])}")
print("C:")
print(f"""B + {itemsetSupport(transactions, [0,4])}, {itemsetSupport(transactions, [3,4])},
""")
print("D:")
print(f"""C +
       {itemsetSupport(transactions, [0,2])}
       {itemsetSupport(transactions, [1,2])},
       {itemsetSupport(transactions, [2,3])},
        {itemsetSupport(transactions, [1,4])},
        {itemsetSupport(transactions, [2,4])},
        {itemsetSupport(transactions, [0,3,4])},
        """)

# Question 12

print("\n-----Question 12------\n")
print("A: ", conf(transactions, [2, 3],[4]))
print("B: ", conf(transactions, [0,4],[3]))
print("C: ", conf(transactions, [0,3],[4]))
print("D: ", conf(transactions, [1,3],[0]))



# Question 13
print("\n-----Question 13------\n")
o1 = tab4[0, :]
o2 = tab4[1, :]
o3 = tab4[2, :]

print(f"A: {Cosine(o1, o2)} > {SMC(o1, o2)}")
print(f"B: {Cosine(o1, o2)} > {Cosine(o1, o3)}")
print(f"C: {Jaccard(o1, o3)} > {SMC(o1, o2)}")
print(f"D: {Jaccard(o1, o3)} > {Cosine(o1, o3)}")

# Question 20

# Question 21

weightsA = np.array([[-1, -1], [1,1], [1,-1]])
weightsB = np.array([[1, -1], [-1,-1], [1,1]])
weightsC = np.array([[1, 1], [-1,-1], [1,-1]])
weightsD = np.array([[-1, -1], [1,1], [-1,1]])

x = np.array([[1, 1], [-1, -1], [1, -1]])
             
print(f"A:\n{class_probabilities(x, weightsA)}")
print(f"B:\n{class_probabilities(x, weightsB)}")
print(f"C:\n{class_probabilities(x, weightsC)}")
print(f"D:\n{class_probabilities(x, weightsD)}")
      
# Question 23 

X = [1, 3, 4, 6, 7, 8, 13, 15, 16, 17]

num_clusters = [4,3,5]
initial_centroids = [[1, 6, 13, 17], [1, 6, 13], [1, 6, 13, 17, 3]]

for centroids,clusters in zip(initial_centroids, num_clusters):
    print(f"K: {clusters}")
    final_clusters, final_centroids = kmeans(X, clusters, centroids)
    print(f"Clusters: {final_clusters}")
    print(f"Centroids: {final_centroids}")



# Question 25

# Setting current weights to intial weights
print("\n------Question 25------\n")
N = 6
currentWeights = np.array([1/N]*N)
labels = [1,1,0,0,1,0]
predictions = [1,1,0,1,0,0]
newWeights = updateWeights(currentWeights, predictions, labels)
print(f"New Weights: {newWeights}")



# Question 26

print("\n------Question 26------\n")
AUC_score = AUC_no_coords(labels, predictions)
print(f"AUC Score: {AUC_score}")