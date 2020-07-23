""" This program is an example of working with DecisionTreeClassifier"""
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
import graphviz
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------------------------
#                       Cancer data - DecisionTreeClassifier
# ----------------------------------------------------------------------------------------------
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
	cancer.data, cancer.target, stratify=cancer.target, random_state=42)

"""Unlimited depth of the tree results in overfitting."""
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


"""
Limiting the depth of the tree to 4 improves result on the test set.
"""
tree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
tree4.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree4.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree4.score(X_test, y_test)))

export_graphviz(tree4, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names,
                impurity=False, filled=True)

with open("tree.dot") as f:
	dot_graph = f.read()
graphviz.Source(dot_graph)


print("Feature importances:\n{}".format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
	n_features = cancer.data.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel("Feature importance")
	plt.ylabel("Feature")
	
plot_feature_importances_cancer(tree)

"""Example """
tree = mglearn.plots.plot_tree_not_monotone()
plt.show()

