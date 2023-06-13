# coding: utf-8

from numpy import *
from matplotlib import pyplot as plt
#from pandas import *
import sys

def loadDataSet(fileName = 'YeastGene_kmeans_cluster.csv'):
    dataMat=[]
    labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArray=line.strip().split(',')
        records = []
        for attr in lineArray[:-1]:
            records.append(float(attr))
        dataMat.append(records)
        labelMat.append(int(lineArray[-1]))
    dataMat = array(dataMat)
    
    labelMat = array(labelMat)
    return dataMat,labelMat

def pca(dataMat, PC_num=2):
    '''
    Input:
        dataMat: obtained from the loadDataSet function, each row represents an observation
                 and each column represents an attribute
        PC_num:  The number of desired dimensions after applying PCA. In this project keep it to 2.
    Output:
        lowDDataMat: the 2-d data after PCA transformation
    '''
    records = len(dataMat)       # number of rows
    X = len(dataMat[0])          # number of attributes/ columns
    
    X_mean = dataMat.mean(axis=0)      # Mean of each column
    
    AdjustedMat = zeros((records, X), dtype=float)      # Initialization with zeros
    
    for i in range(0, records):
        AdjustedMat[i] = dataMat[i] - X_mean     # Mean Adjusted Values X'
        
    CovMat = cov(transpose(AdjustedMat), bias=True)     # Covariance Matrix
    # print (CovMat)
    EigenValues, EigenVectors = linalg.eig(CovMat)      # Eigen Values and Eigen Vectors
    
    i = argsort(EigenValues)[::-1]
    # Sorting the Eigen Values in Descending order (Largest Eigenvalues are used to find the direction of maximum variability)
    # EigenValues = EigenValues[i]
    
    # Sorting the Eigenvectors in the descending order with respect to their Eigenvalues
    EigenVectors = EigenVectors[:,i]
    RequiredEigenVectors = EigenVectors[:,0:PC_num]             # Subset of Eigenvectors for the largest two Eigenvalues (PC_num = 2)
    lowDDataMat = matmul(AdjustedMat, RequiredEigenVectors)     # Low Dimensional Data Matrix is returned after PCA
    
    return array(lowDDataMat)



def plot(lowDDataMat, labelMat, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    x = lowDDataMat[:,0]
    y = lowDDataMat[:,1]
    colors = ['c','g', 'y', 'b', 'r', 'm', 'k']
    # 7 Labels in iyer.csv
    
    LabelColors = empty((len(labelMat)), dtype=str)
    
    i = 0    # Counter
    while (i < len(labelMat)):
        LabelColors[i] = colors[labelMat[i]-1]
        i = i+1
    
    plt.scatter(x,y, c=LabelColors, marker = '.')
    plt.title("Principal Component Analysis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = 'YeastGene_kmeans_cluster.csv'
    figname = filename
    figname = figname.replace('csv','jpg')
    dataMat, labelMat = loadDataSet(filename)
    lowDDataMat = pca(dataMat)
    plot(lowDDataMat, labelMat, figname)