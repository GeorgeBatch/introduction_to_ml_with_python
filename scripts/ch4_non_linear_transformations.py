"""
This program shows example of using non-linear transformations.

Most models work best when each feature (and in regression also the target)
is loosely Gaussian distributed—that is, a histogram of each feature
should have something resembling the familiar “bell curve” shape.
"""

###############################################################################
# -----------------------------------------------------------------------------
#                           Importing packages.
# -----------------------------------------------------------------------------
###############################################################################

# import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import IPython
import mglearn
import sklearn

# data
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

# splitting for ML
from sklearn.model_selection import train_test_split

# pre-processing and scaling
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

# dimensionality reduction and feature extraction

# models
from sklearn.linear_model import Ridge



# model evaluation


###############################################################################
# -----------------------------------------------------------------------------
#                   Creating synthetic data set.
# -----------------------------------------------------------------------------
###############################################################################

rnd = np.random.RandomState(0)

X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)
X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

print("Number of feature appearances:\n{}".format(np.bincount(X[:, 0])))

bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='w')
plt.ylabel("Number of appearances")
plt.xlabel("Value")


###############################################################################
# -----------------------------------------------------------------------------
#                               Models:
# -----------------------------------------------------------------------------
###############################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test score: {:.3f}".format(score))

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("Test score: {:.3f}".format(score))
