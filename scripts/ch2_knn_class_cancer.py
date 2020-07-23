"""
This program shows example of KNN algorithm working with "Breast Cancer" data set.
"""

# ----------------------------------------------------------------------------------------------
#                                       Importing packages.
# ----------------------------------------------------------------------------------------------

import sys
import pandas as pd
import mglearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# ----------------------------------------------------------------------------------------------

cancer = load_breast_cancer()

print("Keys of cancer: \n{}".format(cancer.keys()))
print(cancer['DESCR'][:193] + "\n...")
print("Target names: {}".format(cancer['target_names']))
print("Feature names: \n{}".format(cancer['feature_names']))
print("Type of data: {}".format(type(cancer['data'])))
print("Shape of data: {}".format(cancer['data'].shape))
print("First five columns of data:\n{}".format(cancer['data'][:5]))
print("Type of target: {}".format(type(cancer['target'])))
print("Shape of target: {}".format(cancer['target'].shape))
print("Target:\n{}".format(cancer['target']))

# ----------------------------------------------------------------------------------------------
#                                       Training and testing:
# ----------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

plt.show()