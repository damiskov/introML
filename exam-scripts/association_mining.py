import numpy as np

"""
Script with functions that, given a dataset, can return the association rules:
- Support of itemsets.
- Confidence of association rules.
"""

def individualSupport(names, transactions):
    """
    Given a list of item names and a list of transactions, returns a dictionary with the support of each item.
    The support of an item is the proportion of transactions in which the item appears.

    Parameters:
        - names: list of strings. The names of the items.
        - transactions: list of lists of integers. Each list of integers represents a transaction. The integers are the indices of the items that appear in the transaction.
    
    Returns:
        - items_supports: dictionary. The keys are the names of the items and the values are the support of each item.
    """
    items_supports={}
    for i in range (len(transactions[0])):
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
        - transactions: list of lists of integers. Each list of integers represents a transaction. The integers are the indices of the items that appear in the transaction.
        - X: list of integers. The indices of the items that form the antecedent of the rule.
        - Y: list of integers. The indices of the items that form the consequent of the rule.
    Returns:
        - confidence: float. The confidence of the rule.
    """
    return round(itemsetSupport(transactions, X+Y)/itemsetSupport(transactions, X), 3)


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



    