from __future__ import print_function
import sys
import itertools



def apriori(dataset, min_support=0.5, verbose=False):
    """Implements the Apriori algorithm.

    The Apriori algorithm will iteratively generate new candidate
    k-itemsets using the frequent (k-1)-itemsets found in the previous
    iteration.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate
        candidate itemsets.

    min_support : float
        The minimum support threshold. Defaults to 0.5.

    Returns
    -------
    F : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.

    References
    ----------
    .. [1] R. Agrawal, R. Srikant, "Fast Algorithms for Mining Association
           Rules", 1994.

    """
    C1 = create_candidates(dataset)
    D = list(map(set, dataset))
    F1, support_data = get_freq(D, C1, min_support, verbose=False) # get frequent 1-itemsets
    F = [F1] # list of frequent itemsets; initialized to frequent 1-itemsets
    k = 2 # the itemset cardinality
    while (len(F[k - 2]) > 0):
        Ck = apriori_gen(F[k-2], k) # generate candidate itemsets
        Fk, supK  = get_freq(D, Ck, min_support) # get frequent itemsets
        support_data.update(supK)# update the support counts to reflect pruning
        F.append(Fk)  # add the frequent k-itemsets to the list of frequent itemsets
        k += 1
    
    # The following part of the code throws a TypeError: unhashable type: 'list' 
    # Since this part is mainly used to print out the results,
    # Professor had given the option to write our own codes to print out the results (when asked on Piazza question @77 )
    #if verbose:
        # Print a list of all the frequent itemsets.
        #for kset in F:
            #for item in kset:
                #print(""                     + "{"                     + "".join(str(i) + ", " for i in iter(item)).rstrip(', ')                     + "}"                     + ":  sup = " + str(round(support_data[item], 3)))

    return F, support_data

def create_candidates(dataset, verbose=False):
    """Creates a list of candidate 1-itemsets from a list of transactions.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    Returns
    -------
    The list of candidate itemsets (c1) passed as a frozenset (a set that is
    immutable and hashable).
    """
    c1 = [] # list of all items in the database of transactions
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()

    # The following part of the code throws a TypeError: unhashable type: 'list' 
    # Since this part is mainly used to print out the results,
    # Professor had given the option to write our own codes to print out the results (when asked on Piazza question @77 )
    #if verbose:
        # Print a list of all the candidate items.
        #print(""             + "{"             + "".join(str(i[0]) + ", " for i in iter(c1)).rstrip(', ')             + "}")

    # Map c1 to a frozenset because it will be the key of a dictionary.
    return list(map(frozenset, c1))

def get_freq(dataset, candidates, min_support, verbose=False):
    """

    This function separates the candidates itemsets into frequent itemset and infrequent itemsets based on the min_support,
	and returns all candidate itemsets that meet a minimum support threshold.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    candidates : frozenset
        The list of candidate itemsets.

    min_support : float
        The minimum support threshold.

    Returns
    -------
    freq_list : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.
    """
    
    support_data = {}      # Dictionary which has support counts for all Candidate Itemsets
    freq_list = []         # The list of frequent itemsets.
  
    for c in candidates: 
        support_data[c] = float(0)     # Initializing all the Support Counts to 0
    
    for transaction in dataset:        # Counting the support of each candidate itemset
        for c in candidates:
            if c.issubset(transaction):
                support_data[c] = support_data[c] + (1 / len(dataset)); 

    for c in candidates:
        if (support_data[c] >= min_support):        # if the support of c is not less than minSupport, it is said to be "frequent"
            c = list(c)
            c.sort()
            freq_list.append(c)
    
    # Printing out the results for Frequent Itemsets
    if freq_list:
        if len(freq_list[0])==1:
            print("Length - 1 Frequent Itemsets: ")
  
    print("\nFrequent Itemsets Count =", len(freq_list))
    print("\nFrequent Itemsets: \n", freq_list)
    print("\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print()
    
    return freq_list, support_data


def apriori_gen(freq_sets, k):
    """Generates candidate itemsets (via the F_k-1 x F_k-1 method).

    This part generates new candidate k-itemsets based on the frequent
    (k-1)-itemsets found in the previous iteration.

    The apriori_gen function performs two operations:
    (1) Generate length k candidate itemsets from length k-1 frequent itemsets
    (2) Prune candidate itemsets containing subsets of length k-1 that are infrequent

    Parameters
    ----------
    freq_sets : list
        The list of frequent (k-1)-itemsets.

    k : integer
        The cardinality of the current itemsets being evaluated.

    Returns
    -------
    candidate_list : list
        The list of candidate itemsets.
    """
    
    candidate_list = []        # The list of candidate itemsets.
    freq_sets1 = freq_sets     # Copy of the freq_sets: Lk-1 - list of frequent (k-1)-itemsets.
        
    candidate_list_before_pruning = []  
        
    # Candidate Set Generation:
    # We generate a List of Candidate Sets, Ck by using two Frequent Itemsets from List (Lk-1) if their first k-2 items are identical.
    # Lk-1 x Lk-1 method
    for L in freq_sets:
        for L1 in freq_sets1:
            if (L != L1):
                if L[:k-2] == L1[:k-2]:
                    C = frozenset(L + L1[k-2:])      # merging two itemsets in Lk-1 if their first k-2 items are identical 
                    if C not in candidate_list_before_pruning: 
                        candidate_list_before_pruning.append(C)


    # Candidate Set Pruning:
    # We remove an itemset C from Ck if any (k-1)-subset of this candidate itemset is not in the frequent itemset list Lk-1
    for Ck in candidate_list_before_pruning:
        freq_sets_allsets= list(map(set, freq_sets))     # List of all frequent itemsets
        f = 0
        Ck_subsets = itertools.combinations(Ck, k-1)     # Finding all (k-1)-subsets of Ck
        
        for subset in Ck_subsets:
            subset = frozenset(subset)
            if subset not in freq_sets_allsets:
                f = 1
                break
            
        if f == 0:
            candidate_list.append(Ck)
                        
    # Printing out the results for Candidate Itemsets 
    print("Length -", k,"Candidate Itemsets: ")
    print("\nCandidate Itemsets Count =", len(candidate_list))
    print("\nCandidate Itemsets: \n", candidate_list)
    print("\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print()
    
    #------------------
    print("Length -", k, "Frequent Itemsets: ")   # This is just the heading for Frequent Itemsets which get printed when get_freq() is called next.
    #------------------
    
    return candidate_list


def loadDataSet(fileName, delim=','):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    return stringArr



def run_apriori(data_path, min_support, verbose=False):
    dataset = loadDataSet(data_path)
    F, support = apriori(dataset, min_support=min_support, verbose=verbose)
    return F, support



def bool_transfer(input):
    ''' Transfer the input to boolean type'''
    input = str(input)
    if input.lower() in ['t', '1', 'true' ]:
        return True
    elif input.lower() in ['f', '0', 'false']:
        return False
    else:
        raise ValueError('Input must be one of {T, t, 1, True, true, F, F, 0, False, false}')


if __name__ == '__main__':
    if len(sys.argv)==3:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv)==4:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]), bool_transfer(sys.argv[3]))
    else:
        raise ValueError('Usage: python apriori_templete.py <data_path> <min_support> <is_verbose>')
    print(F)
    print(support)

    '''
    Example: 
    
    python apriori_templete.py market_data_transaction.txt 0.5 
    
    python apriori_templete.py market_data_transaction.txt 0.5 True
    
    '''
