
# coding: utf-8

import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance
import copy
import csv


def loadDataSet(fileName):      # general function to parse tab -delimited floats
    dataMat = []                # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine))       # map all elements to float()
        dataMat.append(fltLine)
    return np.array(dataMat)


def merge_cluster(distance_matrix, cluster_candidate, T):
    ''' Merge two closest clusters according to min distances
    1. Find the smallest entry in the distance matrix—suppose the entry 
        is i-th row and j-th column
    2. Merge the clusters that correspond to the i-th row and j-th column 
        of the distance matrix as a new cluster with index T

    Parameters:
    ------------
    distance_matrix : 2-D array
        distance matrix
    cluster_candidate : dictionary
        key is the cluster id, value is point ids in the cluster
    T: int
        current cluster index

    Returns:
    ------------
    cluster_candidate: dictionary
        upadted cluster dictionary after merging two clusters
        key is the cluster id, value is point ids in the cluster
    merge_list : list of tuples
        records the two old clusters' id and points that have just been merged.
        [(cluster_one_id, point_ids_in_cluster_one), 
         (cluster_two_id, point_ids_in_cluster_two)]
         
         
    Implement Hierarchical clustering algorithm (with Min as inter-cluster distance definition):
        • Obtain the distance matrix by computing Euclidean distance between each pair of objects
        • Let each object be a cluster (Assign the cluster index as 1 to N, where N is the number of objects)
        • Set the current index as T = N+1
        • Repeat
            ▪ Find the smallest entry in the distance matrix—suppose the entry is i-th
              row and j-th column
            ▪ Merge the clusters that correspond to the i-th row and j-th column of the
              distance matrix as a new cluster with index T
            ▪ Remove the rows and columns of the two old clusters and add new row
              and column for the new cluster to the distance matrix by computing the
              distance between the new cluster and each of the remaining clusters
            ▪ T=T+1
        • Until only one cluster remains
        
    '''
    
    
    merge_list = []
    
    # Finding Minimum Value (Smallest entry) in the Distance Matrix (currently)
    min_value = np.inf
    for i in range(len(distance_matrix)):
        for j in range(i):
            if( distance_matrix[i][j] < min_value and i != j):
                min_value = distance_matrix[i][j]
                min_i = i
                min_j = j
                
    
    # min_i => which cluster in cluster_candidate? min_j => which cluster in cluster_candidate?
    # print(min_i,min_j)
    cluster_i = 0
    cluster_j = 0
    
    for k in cluster_candidate:
        values = cluster_candidate[k]
        # number of times element exists in list
        exist_count = values.count(min_i)
        # checking if it is more than 0
        if exist_count > 0:
            cluster_i = k
            
    for l in cluster_candidate:
        values2 = cluster_candidate[l]
        # number of times element exists in list
        exist_count2 = values2.count(min_j)
        # checking if it is more than 0
        if exist_count2 > 0:
            cluster_j = l
    
    #----------------------------------------------------------------------
    # Finding merge_list
    # Extracting specific keys from dictionary
    res = dict((k, cluster_candidate[k]) for k in [cluster_i, cluster_j]
           if k in cluster_candidate)
    
 
    # Converting into list of tuples
    merge_list = [(k, v) for k, v in res.items()]
 
    # Printing list of tuple
    # print(merge_list)
    
    #----------------------------------------------------------------------
    # print(cluster_i, cluster_j)
    
    # print(merge_list)
    
    # Pop Clusters having i and j from cluster_candidate dict
    cluster_candidate.pop(cluster_i)
    cluster_candidate.pop(cluster_j)
    
    # Add a new entry with cluster index T which has value: list of indexes of data points in merged cluster
    merge_list_temp = merge_list
    
    l1 = merge_list_temp[0][1]
    l2 = merge_list_temp[1][1]
    new_list = l2 + l1               # Concatenates l1 to l2
    
    
    # Now we need to add the new_list to cluster_candidate dict with cluster index as T
    
    cluster_candidate[T] = new_list
    
    # print(cluster_candidate)
    
    return cluster_candidate, merge_list


def update_distance(distance_matrix, cluster_candidate, merge_list):
    
    ''' Update the distance matrix
    
    Parameters:
    ------------
    distance_matrix : 2-D array
        distance matrix
    cluster_candidate : dictionary
        key is the updated cluster id, value is a list of point ids in the cluster
    merge_list : list of tuples
        records the two old clusters' id and points that have just been merged.
        [(cluster_one_id, point_ids_in_cluster_one), 
         (cluster_two_id, point_ids_in_cluster_two)]

    Returns:
    ------------
    distance_matrix: 2-D array
        updated distance matrix       
    '''
    
    
    # TODO
    
    merge_list_temp = merge_list
    
    tuple1 = merge_list_temp[0]
    tuple2 = merge_list_temp[1]
    
    #l1 = list(tuple1)
    #l1.pop(0)
    
    #l2 = list(tuple2)
    #l2.pop(0)
    
    l1 = tuple1[1]
    l2 = tuple2[1]
    l = l1 + l2
    
    for i in l:
        for j in l:
            distance_matrix[i][j] = 100000
    
    for i in l1:
        for j in l1:
            distance_matrix[i][j] = 100000
            
    for i in l2:
        for j in l2:
            distance_matrix[i][j] = 100000 
    
    for i in l1:
        for j in l2:
            distance_matrix[i][j] = 100000
    
    return distance_matrix


def agglomerative_with_min(data, cluster_number):
    """
    agglomerative clustering algorithm with min link

    Parameters:
    ------------
    data : 2-D array
        each row represents an observation and 
        each column represents an attribute

    cluster_number : int
        number of clusters

    Returns:
    ------------
    clusterAssment: list
        assigned cluster id for each data point
    """
    cluster_candidate = {}
    N = len(data)
    # initialize cluster, each sample is a single cluster at the beginning
    for i in range(N):
        cluster_candidate[i+1] = [i]  # key: cluser id; value: point ids in the cluster

    # initialize distance matrix
    distance_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j == i: # or j<=i
                distance_matrix[i,j] = 100000
            else:
                distance_matrix[i,j] = np.sqrt(np.sum((data[i]-data[j])**2))
    
    # hiearchical clustering loop
    T = N + 1 # cluster index
    for i in range(N - cluster_number):
        cluster_candidate, merge_list = merge_cluster(distance_matrix, cluster_candidate, T)
        distance_matrix   =  update_distance(distance_matrix, cluster_candidate, merge_list )
        print('%d-th merging: %d, %d, %d'% (i, merge_list[0][0], merge_list[1][0], T))
        T += 1
        # print(cluster_candidate)


    # assign new cluster id to each data point 
    clusterAssment = [-1] * N
    for cluster_index, cluster in enumerate(cluster_candidate.values()):
        for c in cluster:
            clusterAssment[c] = cluster_index
    # print (clusterAssment)
    return clusterAssment


def saveData(save_filename, data, clusterAssment):
    clusterAssment = np.array(clusterAssment, dtype = object)[:,None]
    data_cluster = np.concatenate((data, clusterAssment), 1)
    data_cluster = data_cluster.tolist()

    with open(save_filename, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data_cluster)
    f.close()



if __name__ == '__main__':
    if len(sys.argv) == 3:
        data_filename = sys.argv[1]
        cluster_number = int(sys.argv[2])
    else:
        data_filename = 'Utilities.csv'
        cluster_number = 1

    save_filename = data_filename.replace('.csv', '_hc_cluster.csv')

    data = loadDataSet(data_filename)

    clusterAssment = agglomerative_with_min(data, cluster_number)

    saveData(save_filename, data, clusterAssment)
    
    
    
    
    #for i in range(len(distance_matrix)):
        #if( i > min_i  and i < min_j ):
            #distance_matrix[i][min_i] = min(distance_matrix[i][min_i],distance_matrix[min_j][i])

        #elif( i > min_j ):
            #distance_matrix[i][min_i] = min(distance_matrix[i][min_i],distance_matrix[i][min_j])

    #for j in range(len(distance_matrix)):
        #if( j < min_i ):
            #distance_matrix[min_i][j] = min(distance_matrix[min_i][j],distance_matrix[min_j][j])

    #remove one of the old clusters data from the distance matrix
    #distance_matrix = np.delete(distance_matrix, min_j, axis=1)
    #distance_matrix = np.delete(distance_matrix, min_j, axis=0)

    #A[min_i] = A[min_i] + A[min_j] 
    #A.pop(min_j)
