"""
This program shows example of using binning and interactions on "housing"
data set.

In this data set we only have one predictor which has a
continuous scale, but it may actually affect very differently on
different intervals.
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
from sklearn.datasets import load_boston

# splitting for ML
from sklearn.model_selection import train_test_split

# dimensionality reduction and feature extraction

# pre-processing and scaling
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

# models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# model evaluation


###############################################################################
# -----------------------------------------------------------------------------
#               Importing and getting to know our data:
# -----------------------------------------------------------------------------
###############################################################################


boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
	boston.data, boston.target, random_state=0)



###############################################################################
# -----------------------------------------------------------------------------
#                   Pre-processing and scaling:
# -----------------------------------------------------------------------------
###############################################################################

# rescale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


###############################################################################
# -----------------------------------------------------------------------------
#               Representing Data and Engineering Features
# -----------------------------------------------------------------------------
###############################################################################

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)

X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_poly.shape: {}".format(X_train_poly.shape))

print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


###############################################################################
# -----------------------------------------------------------------------------
#                               Models
# -----------------------------------------------------------------------------
###############################################################################

"""
Here Ridge regression and Random forests will be compared.
"""

# -----------------------------------------------------------------------------
#                           Ridge regression:
# -----------------------------------------------------------------------------

"""
Clearly, the interactions and polynomial features gave us a good boost 
in performance when using Ridge. When using a more complex model like 
a random forest, the story is a bit different, though:
"""

ridge = Ridge().fit(X_train_scaled, y_train)
print("Score without interactions: {:.3f}".format(
	ridge.score(X_test_scaled, y_test)))

ridge = Ridge().fit(X_train_poly, y_train)
print("Score with interactions: {:.3f}".format(
	ridge.score(X_test_poly, y_test)))


# -----------------------------------------------------------------------------
#                           Random forest:
# -----------------------------------------------------------------------------

"""
You can see that even without additional features, the random forest
beats the performance of Ridge. Adding interactions and polynomials
actually decreases performance slightly.
"""

rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print("Score without interactions: {:.3f}".format(
	rf.score(X_test_scaled, y_test)))

rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print("Score with interactions: {:.3f}".format(rf.score(X_test_poly, y_test)))
