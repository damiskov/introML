import numpy as np

def SMC(a, b):
    return np.round(np.sum(a == b) / len(a), 3)

def Jaccard(a, b):
    return np.round(np.sum(a & b) / np.sum(a | b), 3)

def Cosine(a, b):
    return np.round(np.dot(a, b)/ (np.linalg.norm(a) * np.linalg.norm(b)), 3)

def extended_Jaccard(a, b):
    return np.round(np.dot(a, b) / np.sum(a | b), 3)

def extended_Cosine(a,b):
    return np.round((np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)), 3)

def pNorm(a, b, p):
    if p == 0:
        raise ValueError("p must be different from 0")
    if p == np.inf:
        return np.max(np.abs(a - b))
    return np.round(np.sum(np.abs(a - b) ** p) ** (1/p), 3)



if __name__=="__main__":
    x35 = np.array([-1.24, -0.26, -1.04])
    x53 = np.array([-0.6, -0.86, -0.5])

    d = x35 - x53
    print('p=1:', pNorm(x35, x53, 1))
    print('p=4:', pNorm(x35, x53, 4))
    print('p=inf:', pNorm(x35, x53, np.inf))
    print('cosine:', Cosine(x35, x53))
    
    

    print(f"""
    Split 1: {1/46 + 8/41}
    Split 2: {45/46 + 33/41}
    Split 3: {0/25  + 77/97}
""")