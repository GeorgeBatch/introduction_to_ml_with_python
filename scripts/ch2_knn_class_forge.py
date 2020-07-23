"""
This program shows example of KNN algorithm working with "forge" data set.
"""

# ----------------------------------------------------------------------------------------------
#                                       Importing packages:
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
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------------------------------
#                                       Generating data:
# ----------------------------------------------------------------------------------------------

# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
# plt.show()


# ----------------------------------------------------------------------------------------------
#                                       Splitting the data for training and testing:
# ----------------------------------------------------------------------------------------------

"""Splitting the data (3:1 for training and testing;
random_state=0 is needed to always have the same split)"""
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))


print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
# ----------------------------------------------------------------------------------------------
#                                       Training and testing for 1 and 3 neighbours:
# ----------------------------------------------------------------------------------------------

# 1 neighbour
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set score (1 neighbour): {:.2f}".format(clf.score(X_test, y_test)))
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

# 3 neighbours
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set score (3 neighbours): {:.2f}".format(clf.score(X_test, y_test)))
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

# ----------------------------------------------------------------------------------------------
#                                       Plotting decision boundaries for 1,3,9 nn:
# ----------------------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

plt.show()