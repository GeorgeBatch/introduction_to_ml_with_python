"""
This program shows example of Linear, Ridge and Lasso regression algorithms working with "Boston housing" data set.
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
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# ----------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# ----------------------------------------------------------------------------------------------

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# using linear regression:
lr = LinearRegression().fit(X_train, y_train)
print("Linear regression:")
print("     Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# ----------------------------------------------------------------------------------------------
#           Ridge regression; Comparing different values of alpha (restricting parameter):
# ----------------------------------------------------------------------------------------------

"""
Greater alpha for Ridge regression means greater restriction (e.g. alpha=10 puts very serious restrictions).
Restrictions close to 0 give similar results to linear regression (here alpha=0.1)
By default alpha=1 us used as an argument of Ridge() 
"""

# here alpha=1 by default
ridge = Ridge().fit(X_train, y_train)
print('Ridge regression, alpha=1:')
print("     Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# we have increased alpha to 10, putting serious restrictions on coefficients
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print('Ridge regression, alpha=10:')
print("     Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# setting alpha=0.1 gives results very similar to linear regression
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print('Ridge regression, alpha=0.1:')
print("     Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

# ----------------------------------------------------------------------------------------------
#           Lasso regression; Comparing different values of alpha (restricting parameter):
# ----------------------------------------------------------------------------------------------

"""
Greater alpha for Lasso regression means greater restriction (e.g. alpha=1).
Restriction close to 0 gives similar results to linear regression (here: alpha=0.0001)
By default alpha=1 us used as an argument of Lasso() 
"""

lasso = Lasso().fit(X_train, y_train)
print('Lasso regression, alpha=1:')
print("     Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("     Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print('Lasso regression, alpha=0.01:')
print("     Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("     Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print('Lasso regression, alpha=0.0001:')
print("     Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("     Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("     Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

# ----------------------------------------------------------------------------------------------
#           Coefficient magnitudes & Learning curves of Linear and Ridge regressions:
# ----------------------------------------------------------------------------------------------
"""
Here we can see the coefficients produced by the models for each of 105 predictors.
For more restricted models the coefficients are closer to 0.
Ridge is L2 restricted, i.e. sum of squared coefficients is penalised.

Learning curve for train set is better for Linear regression.
However, for test set Ridge works better.
"""

# plotting coefficient magnitudes against coefficient indexes
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()

# plotting a learning curve
mglearn.plots.plot_ridge_n_samples()
plt.show()

# ----------------------------------------------------------------------------------------------
#           Coefficient magnitudes & Learning curves of Ridge and Lasso regressions:
# ----------------------------------------------------------------------------------------------
"""
Here we can see the coefficients produced by the models for each of 105 predictors.
For more restricted models the coefficients are closer to 0.

Lasso is L1 restricted, i.e. sum of absolute values of the coefficients is penalised.
Tha latter forces some coefficients to be zero for Lasso, making the model simpler. While for ridge
regression all coefficients are nonzero.
"""

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

plt.show()
