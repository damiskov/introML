import numpy as np
from comparing_partitions import countingMat, index

print("\n-----Question 19------\n")

Z = np.array(8*[0]+3*[1])
Q = np.array(8*[0]+2*[1]+[0])

print(f"Jaccard Index: {index(Z, Q, 'jaccard')}")
