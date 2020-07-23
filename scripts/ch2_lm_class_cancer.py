"""
This program shows example of Logistic regression and SVM algorithms working with "Breast cancer" data set.
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
from sklearn.datasets import load_breast_cancer

# ----------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# ----------------------------------------------------------------------------------------------
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# ----------------------------------------------------------------------------------------------
#           Logistic regression; Comparing different values of alpha (restricting parameter):
# ----------------------------------------------------------------------------------------------

# C=1 by default - gives very close train/test results, so we are probably underfitting
logreg = LogisticRegression().fit(X_train, y_train)
print('Logistic regression; C=1')
print('     Training set score: {:.3f}'.format(logreg.score(X_train, y_train)))
print('     Test set score: {:.3f}'.format(logreg.score(X_test, y_test)))

# C=100 - gives better results (might be a bit overfitting)
print('Logistic regression; C=100')
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print('     Training set score: {:.3f}'.format(logreg100.score(X_train, y_train)))
print('     Test set score: {:.3f}'.format(logreg100.score(X_test, y_test)))

# C=0.01 results in underfitting
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print('Logistic regression; C=0.01')
print('     Training set score: {:.3f}'.format(logreg001.score(X_train, y_train)))
print('     Test set score: {:.3f}'.format(logreg001.score(X_test, y_test)))

plt.plot(logreg.coef_.T, 'o', label='C=1')
plt.plot(logreg100.coef_.T, '^', label='C=100')
plt.plot(logreg001.coef_.T, 'v', label='C=0.001')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.legend()

plt.show()


# L1 regularisation reduces the number of parameters and C=100 gives even better test-train scores pair
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
	lr_l1 = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)
	print('Training accuracy of l1 logreg with C={:.3f}: {:.2f}'.format(
		C, lr_l1.score(X_train, y_train)))
	print('Test accuracy of l1 logreg with C={:.3f}: {:.2f}'.format(
		C, lr_l1.score(X_test, y_test)))
	plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')

plt.ylim(-5, 5)
plt.legend(loc=3)

plt.show()