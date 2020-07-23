"""
This program shows example of using binning and interactions on "wave"
data set. In this data set we only have one predictor which has a
continuous scale, but it may actually affect very differently on
different intervals.
"""

###############################################################################
# -----------------------------------------------------------------------------
#                                       Importing packages.
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


# model evaluation


###############################################################################
# -----------------------------------------------------------------------------
#               Importing and getting to know our data:
# -----------------------------------------------------------------------------
###############################################################################

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)



###############################################################################
# -----------------------------------------------------------------------------
#                           Regression models:
# -----------------------------------------------------------------------------
###############################################################################

# ----------------------------------------------------------------------
#                   Fitting regression models:
# ----------------------------------------------------------------------

# fitting DecisionTreeRegressor
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="decision tree")

# fitting LinearRegression
reg = LinearRegression().fit(X, y)

# ----------------------------------------------------------------------
#                   Evaluating regression models:
# ----------------------------------------------------------------------
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

plt.show()
# ----------------------------------------------------------------------
#                   Binning (cts var -> categorical)
# ----------------------------------------------------------------------
"""
Here we will turn it into binned categorical variables, assuming that 
data behaves uniformly within
each bin.
"""

# creating bins
bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))

# identifying which bin the data point belongs to
which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print(X_binned[:5])


# now, as we have X_binned ()
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")

plt.show()

# ----------------------------------------------------------------------
#                   Binning with original slope
# ----------------------------------------------------------------------

"""
Now we will add same slope to each bin. The bin itself will determine 
the intercept (the offset).
"""

X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)
line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='linear regression combined')

for bin in bins:
	plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:, 0], y, 'o', c='k')

plt.show()

# ----------------------------------------------------------------------
#                   Binning with different slopes.
# ----------------------------------------------------------------------

print('X.shape = {}'.format(X.shape))
print('X_binned.shape = {}'.format(X_binned.shape))
print('(X * X_binned).shape = {}'.format((X * X_binned).shape))

X_product = np.hstack([X_binned, X * X_binned])
print('X_product.shape'.format(X_product.shape))

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')

for bin in bins:
	plt.plot([bin, bin], [-3, 3], ':', c='k')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

plt.show()

# ----------------------------------------------------------------------
#                       Polynomial features
# ----------------------------------------------------------------------
"""
include polynomials up to x ** 10:
the default "include_bias=True" adds a feature that's constantly 1
?: Is it to give model an intercept?

Careful! Polynomial regression models are very sensitive to regions with
low data density.
"""


poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

print("X_poly.shape: {}".format(X_poly.shape))

print("Entries of X:\n{}".format(X[:5]))
print("Entries of X_poly:\n{}".format(X_poly[:5]))

print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


# Using polynomial features together with a linear regression model
# yields the classical model of polynomial regression.

reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

plt.show()


# ----------------------------------------------------------------------
#                   SVM vs poly regression:
# ----------------------------------------------------------------------

for gamma in [1, 10]:
	svr = SVR(gamma=gamma).fit(X, y)
	plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

plt.show()