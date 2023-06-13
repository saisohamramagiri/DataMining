import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


import seaborn as sns
#matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('glass.csv')
#print(data)

X = data.drop('Type', axis = 1)
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNeighborsClassifier()

param_dist1 = {'n_neighbors' : [i for i in range(2, 25)],
             'weights' : ['uniform','distance'],
             'metric' : ['minkowski', 'manhattan']}
rcv1 = RandomizedSearchCV(knn, param_distributions = param_dist1, n_iter = 10, cv = 5)
rcv1.fit(X, y)
print(rcv1.best_score_)
print(rcv1.best_params_)
knn_final = KNeighborsClassifier(n_neighbors = 7,
                                weights = 'uniform', metric = 'manhattan').fit(X_train, y_train)

print(knn_final.score(X_train, y_train), knn_final.score(X_test, y_test))


knn_final = KNeighborsClassifier(n_neighbors = 7,
                                weights = 'uniform', metric = 'manhattan').fit(X_train, y_train)





svm = SVC()
param_dist2 = {'C' : [1, 10, 100, 1E3, 1E4],
              'kernel': ['rbf']}
rcv2 = RandomizedSearchCV(svm, param_distributions = param_dist2, n_iter = 5, cv =5)
rcv2.fit(X,y)
print("\n\n")
print(rcv2.best_estimator_)
print(rcv2.best_params_)
print(rcv2.best_score_)

svm_final = SVC(C = 1E3, kernel = 'rbf').fit(X_train, y_train)
print(svm_final.score(X_train, y_train), svm_final.score(X_test, y_test))


nb = GaussianNB()
print("\n\n")
nb.fit(X_train, y_train)
#Make prediction on X_test
#y_pred_NB=nb.predict(X_test)
#conf_mat_NB=confusion_matrix(y_test, y_pred_NB)
#plt.figure(figsize=(10,8))
#sns.heatmap(conf_mat_NB,annot=True,fmt='d')

#naive_acc=accuracy_score(y_test,y_pred_NB)
print(nb.score(X_train, y_train), nb.score(X_test, y_test))



dt = DecisionTreeClassifier()
param_dist3 = {'max_depth': [None, 2, 3, 4, 5, 6],
             'criterion': ['gini', 'entropy'],
             'min_samples_split': [5, 10, 12],
             'max_leaf_nodes': [10, 15, 20, None]}

rcv3 = RandomizedSearchCV(dt, param_distributions = param_dist3, n_iter = 10, cv = 5)
rcv3.fit(X, y)

print("\n\n")
print(rcv3.best_estimator_)
print(rcv3.best_params_)
print(rcv3.best_score_)

dt_final = DecisionTreeClassifier(criterion = 'gini', max_leaf_nodes = 10,
                                 max_depth = 4, min_samples_split = 2)
dt_final.fit(X_train, y_train)

print(dt_final.score(X_train, y_train), dt_final.score(X_test, y_test))



rf = RandomForestClassifier()
#Cross Validation for Random Forest
param_dist4 = {'n_estimators' : [100, 120, 150, 200], 'criterion': ['gini', 'entropy']}
rcv4 = RandomizedSearchCV(rf, param_distributions = param_dist4, n_iter = 6, cv = 8)
rcv4.fit(X, y)
print("\n\n")
print(rcv4.best_params_)
print(rcv4.best_score_)
rf_final = RandomForestClassifier(50, max_depth = 5, random_state = 20)
rf_final.fit(X_train, y_train)
print(rf_final.score(X_train, y_train), rf_final.score(X_test, y_test))


bc = BaggingClassifier(DecisionTreeClassifier())
bc.fit(X_train, y_train)

print("\n\n")
print(bc.score(X_train, y_train), bc.score(X_test, y_test))


adb = AdaBoostClassifier(base_estimator = RandomForestClassifier(),n_estimators = 100, learning_rate = 1, random_state = 52)
adb.fit(X_train, y_train)

print(adb.score(X_train, y_train), adb.score(X_test, y_test))
