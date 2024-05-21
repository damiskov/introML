import numpy as np
import numpy as np
from itertools import combinations
from collections import defaultdict

"""
Script with functions that, given a dataset, can return the association rules:
- Support of itemsets.
- Confidence of association rules.
"""

def individualSupport(transactions):
    """
    Given a list of transactions, returns a dictionary with the support of each item.
    The support of an item is the proportion of transactions in which the item appears.

    Parameters:
        - names: list of strings. The names of the items.
        - transactions: list of lists of integers. Each list of integers represents a transaction. The integers are the indices of the items that appear in the transaction.
    
    Returns:
        - items_supports: dictionary. The keys are the names of the items and the values are the support of each item.
    """
    names = [f"f"+str(i+1) for i in range(transactions.shape[1])]
    print(names)
    items_supports={}
    for i in range (transactions.shape[1]):
        value = sum([t[i] for t in transactions])/len(transactions)
        items_supports[names[i]]=round(value, 3)
    return items_supports

def itemsetSupport(transactions, indices):
    """
    Given a list of transactions and a list of indices, returns the support of the itemset represented by the indices.

    Parameters:
        - item_names: nparray of strings. The names of the items.
        - transactions: list of lists of integers. Each list of integers represents a transaction. The integers are the indices of the items that appear in the transaction.
        - indices: list of integers. The indices of the items that form the itemset.
    Returns:
        - support: float. The support of the itemset.
    """
    support = 0
    for t in transactions:
        boolean_rep = np.prod(np.array([t[i]==1 for i in indices]))
        if boolean_rep:
            support += 1
    support /= len(transactions)

    
    return round(support, 3)

def conf(transactions, X,Y):
    """
    Calculates confidence of association rule X->Y
    Parameters:
        X AND Y MUST BE LISTS (not nparrays!)
        - transactions: list of lists of integers. Each list of integers represents a transaction. The integers are the indices of the items that appear in the transaction.
        - X: list of integers. The indices of the items that form the antecedent of the rule.
        - Y: list of integers. The indices of the items that form the consequent of the rule.
    Returns:
        - confidence: float. The confidence of the rule.
    """
    return round(itemsetSupport(transactions, X+Y)/itemsetSupport(transactions, X), 3)

def apriori(transactions, epsilon):
    num_transactions, num_items = transactions.shape
    
    # Function to calculate support
    def get_support(itemset):
        return np.sum(np.all(transactions[:, list(itemset)] == 1, axis=1))
    
    # Generate C1
    C1 = {frozenset([i]): get_support([i]) for i in range(num_items)}
    print("C1:", C1)
    
    # Prune C1 to get L1
    L1 = {itemset: support for itemset, support in C1.items() if support >= epsilon}
    print("L1:", L1)
    
    # Initialize previous frequent itemsets and the final result
    L_prev = L1
    all_frequent_itemsets = L1.copy()
    k = 2
    
    while L_prev:
        # Generate Ck from L(k-1)
        Ck = defaultdict(int)
        L_prev_itemsets = list(L_prev.keys())
        
        for i in range(len(L_prev_itemsets)):
            for j in range(i+1, len(L_prev_itemsets)):
                # Generate the union of two itemsets if their first k-2 items are equal
                l1 = list(L_prev_itemsets[i])
                l2 = list(L_prev_itemsets[j])
                if l1[:-1] == l2[:-1]:
                    candidate = frozenset(l1) | frozenset(l2)
                    if len(candidate) == k:
                        Ck[candidate] = get_support(candidate)
        
        print(f"C{k}:", dict(Ck))
        
        # Prune Ck to get Lk
        Lk = {itemset: support for itemset, support in Ck.items() if support >= epsilon}
        print(f"L{k}:", Lk)
        
        if not Lk:
            break
        
        L_prev = Lk
        all_frequent_itemsets.update(Lk)
        k += 1
    
    return all_frequent_itemsets




if __name__=="__main__":
 
    transactions = ["1 1 1 0 0",
    "1 1 1 0 0",
    "1 1 1 0 0",
    "1 1 1 0 0",
    "1 1 1 0 0",
    "0 1 1 0 0",  
    "0 1 0 1 1",  
    "1 1 1 0 0", 
    "1 0 1 0 0", 
    "0 0 0 1 1", 
    "0 1 0 1 1"]

    transactions = np.array([e.split(' ') for e in transactions], dtype=int)

    items_names = "f1 f2 f3 f4 f5".split()

    print("Question 20")
    print("A")
    print(individualSupport(items_names, transactions))
    print("B")
    print(itemsetSupport(transactions, [0,1]))
    print(itemsetSupport(transactions, [0,2]))
    print(itemsetSupport(transactions, [1,2]))
    print("C")
    print(itemsetSupport(transactions, [0,1,2]))

    print("Question 21")

    print(itemsetSupport(transactions, [0,1,2]))
    print(itemsetSupport(transactions, [0,1]))

    print(conf(transactions, [0, 1],[2]))



    
