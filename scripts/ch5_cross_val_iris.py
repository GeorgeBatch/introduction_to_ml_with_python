"""
This program shows example of using cross-validation on "iris" data set.
"""

# ----------------------------------------------------------------------------------------------
#                                       Importing packages.
# ----------------------------------------------------------------------------------------------

# import sys
# import pandas as pd
# import mglearn
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy as sp
# import IPython
# import mglearn
# import sklearn

# data
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

# splitting for ML
from sklearn.model_selection import train_test_split

# pre-processing and scaling

# dimensionality reduction

# models
from sklearn.linear_model import LogisticRegression

# model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit

# ----------------------------------------------------------------------------------------------
#                          Importing and getting to know our data:
# ----------------------------------------------------------------------------------------------

iris = load_iris()

print("Iris labels:\n{}".format(iris.target))

# ----------------------------------------------------------------------------------------------
#                           Fitting Logistic regression model:
# ----------------------------------------------------------------------------------------------
logreg = LogisticRegression()

################################################################################################
# ----------------------------------------------------------------------------------------------
#                           Evaluating Logistic regression model:
# ----------------------------------------------------------------------------------------------
################################################################################################


# ----------------------------------------------------------------------------------------------
#                                   Using: cross_val_score
# ----------------------------------------------------------------------------------------------
"""
For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, 
StratifiedKFold is used. In all other cases, KFold is used.

So here  StratifiedKFold is used and gives good results
"""

scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("Stratified 3-fold Cross-validation scores: {}\n".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Stratified 5-fold Cross-validation scores: {}\n".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


# ----------------------------------------------------------------------------------------------
#                       Non-stratified KFold (bad idea) cross_val_score
# ----------------------------------------------------------------------------------------------
"""
Here there are 2 examples by means of which we can verify that it is indeed a really bad idea to 
use three-fold (nonstratified) cross-validation on the iris dataset:
"""
# mglearn.plots.plot_cross_validation()

kfold = KFold(n_splits=5)
print("Nonstratified 5-fold Cross-validation scores:\n{}\n".format(
	cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

kfold = KFold(n_splits=3)
print("Nonstratified 3-fold Cross-validation scores:\n{}\n".format(
	cross_val_score(logreg, iris.data, iris.target, cv=kfold)))


# ----------------------------------------------------------------------------------------------
#                   Shuffled KFold (instead of stratifying) cross_val_score
# ----------------------------------------------------------------------------------------------
# mglearn.plots.plot_stratified_cross_validation()

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("Shuffled 3-fold Cross-validation scores:\n{}\n".format(
	cross_val_score(logreg, iris.data, iris.target, cv=kfold)))


# ----------------------------------------------------------------------------------------------
#                           Leave-one-out cross_val_score
# ----------------------------------------------------------------------------------------------
"""
Another frequently used cross-validation method is leave-one-out. 
You can think of leave-one-out cross-validation as k-fold cross-validation where each fold 
is a single sample. For each split, you pick a single data point to be the test set. 
This can be very time consuming, particularly for large datasets, but sometimes provides better 
estimates on small datasets:
"""
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Leave-one-out mean accuracy: {:.2f}\n".format(scores.mean()))


# ----------------------------------------------------------------------------------------------
#                           Shuffle-split cross-validation
# ----------------------------------------------------------------------------------------------
"""
Another, very flexible strategy for cross-validation is shuffle-split cross-validation. 
In shuffle-split cross-validation, each split samples train_size many points for the training 
set and test_size many (disjoint) point for the test set. This splitting is repeated n_iter times.


There is also a stratified variant of ShuffleSplit, aptly named StratifiedShuffleSplit, 
which can provide more reliable results for classification tasks.
"""
# mglearn.plots.plot_shuffle_split()



shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=8)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Shuffle-split cross-validation scores:\n{}\n".format(scores))


# ----------------------------------------------------------------------------------------------
#                           Cross-validation with groups
# ----------------------------------------------------------------------------------------------
"""
If data can be put into groups, it will be easier for the algorithm to classify an item
in the test set, if another item from the same group has already been there.

This can affect our model evaluation. To prevent this Cross-validation with groups can be used.
Then each group is either entirely in the training set or entirely in the test set.

Note: groups should not be confused with labels (the classification classes)!!!

"""
# mglearn.plots.plot_label_kfold()


from sklearn.model_selection import GroupKFold

# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)

# assume the first three samples belong to the same group,
# then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("Cross-validation with groups scores:\n{}".format(scores))


# ----------------------------------------------------------------------------------------------
"""
There are more splitting strategies for cross-validation in scikit-learn, 
which allow for an even greater variety of use cases (you can find these in the scikit-learn user guide). 
However, the standard KFold, StratifiedKFold, and GroupKFold are by far the most commonly used ones.
"""
