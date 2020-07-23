"""
This program shows example of using cross-validation on "Blobs" data set.
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
import mglearn
import sklearn

# data
from sklearn.datasets import make_blobs

# splitting for ML
from sklearn.model_selection import train_test_split

# pre-processing and scaling

# dimensionality reduction


# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# ----------------------------------------------------------------------------------------------

# create a synthetic dataset
X, y = make_blobs(random_state=0)

# split data and labels into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate a model and fit it to the training set
logreg = LogisticRegression().fit(X_train, y_train)
# evaluate the model on the test set
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))

mglearn.plots.plot_cross_validation()
plt.show()