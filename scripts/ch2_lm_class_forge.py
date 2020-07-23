"""
This program shows example of Logistic regression and SVM algorithms working with "forge" data set.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# ----------------------------------------------------------------------------------------------
"""
For LogisticRegression and LinearSVC the trade-off parameter that determines the strength of the regularization
is called C, and higher values of C correspond to less regularization. In other words, when you use a high value for 
the parameter C, LogisticRegression and LinearSVC try to fit the training set as best as possible,
while with low values of the parameter C, the models put more emphasis on finding a coefficient vector (w)
that is close to zero.

There is another interesting aspect of how the parameter C acts.
Using low values of C will cause the algorithms to try to adjust to the “majority” of data points, 
while using a higher value of C stresses the importance that each individual data point be classified correctly.
"""
X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	clf = model.fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title("{}".format(clf.__class__.__name__))
	ax.set_xlabel("Feature 0")
	ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()

mglearn.plots.plot_linear_svc_regularization()
plt.show()

