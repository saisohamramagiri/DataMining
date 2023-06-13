
# coding: utf-8

import sys
from numpy import *
#from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np
import copy
import csv


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return mat(dataMat)


def loadCenterSet(fileName):      #general function to parse tab -delimited floats
    centerMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        centerMat.append(fltLine)
    return mat(centerMat)


def assignCluster(dataSet, k, centroids):
    '''For each data point, assign it to the closest centroid
    Inputs:
        dataSet: each row represents an observation and 
                 each column represents an attribute
        k:  number of clusters
        centroids: initial centroids or centroids of last iteration
    Output:
        clusterAssment: list
            assigned cluster id for each data point
            
            
    Implement K-means algorithm as follows:
             Repeat T times:
                • For each object xi
                    ▪ Calculate Euclidean distance between xi and each of the K centroids
                    ▪ Assign xi to the cluster whose centroid is the closest to xi
                • For each cluster
                    ▪ Calculate its centroid as the mean of all the objects in that cluster
    '''
    #TODO
    
    clusterAssment = []     # List for assigned cluster id for each data point

    for each_object_X in dataSet:
        
        # Initialization of required variables
        min_dist = 1000000          # A very high value
        index_counter = 0
        
        Xi = -1         # Index of the datapoint Xi which is closest to the cluster (whose centroid is known)
        
        for each_centroid in centroids:
            
            dist = distance.euclidean(each_object_X, each_centroid)
            
            if dist < min_dist:
                Xi = index_counter 
                min_dist = dist
                
            index_counter = index_counter + 1
            
        clusterAssment.append(Xi)

    return clusterAssment


def getCentroid(dataSet, k, clusterAssment):
    '''Recalculating the Centroids:
    Input: 
        dataSet: each row represents an observation and 
            each column represents an attribute
        k:  number of clusters
        clusterAssment: list
            assigned cluster id for each data point
    Output:
        centroids: cluster centroids
    '''
    #TODO
    centroids = []       # List Initialization
    
    for each_cluster in range(0, k):
        
        centr = []
        
        for j in range(0, len(clusterAssment)):
            if clusterAssment[j] == each_cluster:
                centr.append(dataSet[j])
                
        centr = np.array(centr)
        
        cluster_centroid = centr.mean(axis = 0)  # The mean of all the objects in that cluster
        
        centroids.append(cluster_centroid.tolist())

    return centroids


def kMeans(dataSet, T, k, centroids):
    '''
    Input:
        dataSet: each row represents an observation and 
                each column represents an attribute
        T:  number of iterations
        k:  number of clusters
        centroids: initial centroids
    Output:
        centroids: final cluster centroids
        clusterAssment: list
            assigned cluster id for each data point
    '''
    clusterAssment = [0] * len(dataSet)
    pre_clusters  = [1] * len(dataSet)

    i=1
    while i < T and list(pre_clusters) != list(clusterAssment):
        pre_clusters = copy.deepcopy(clusterAssment) 
        clusterAssment = assignCluster(dataSet, k, centroids )
        centroids      = getCentroid(dataSet, k, clusterAssment)
        i=i+1

    return centroids, clusterAssment


def saveData(save_filename, data, clusterAssment):
    clusterAssment = np.array(clusterAssment, dtype = object)[:,None]
    data_cluster = np.concatenate((data, clusterAssment), 1)
    data_cluster = data_cluster.tolist()

    with open(save_filename, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data_cluster)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        data_filename = sys.argv[1]
        centroid_filename = sys.argv[2]
        k = int(sys.argv[3])
    else:
        data_filename = 'YeastGene.csv'
        centroid_filename = 'YeastGene_Initial_Centroids.csv'
        k = 6

    save_filename = data_filename.replace('.csv', '_kmeans_cluster.csv')

    data = loadDataSet(data_filename)
    centroids = loadCenterSet(centroid_filename)
    centroids, clusterAssment = kMeans(data, 7, k, centroids )
    print(centroids)
    saveData(save_filename, data, clusterAssment)


    ### Example: python kmeans_template.py Iris.csv Iris_Initial_Centroids.csv
    
    
     ### Usage: python kmeans_template.py YeastGene.csv YeastGene_Initial_Centroids.csv
