"""
This program is the first machine learning experience.
Here I am working with Iris data set.
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
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------------------------------
#                                       Exploring the data
# ----------------------------------------------------------------------------------------------

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

# ----------------------------------------------------------------------------------------------
#                                       Splitting the data for training and testing:
# ----------------------------------------------------------------------------------------------

"""Splitting the data (3:1 for training and testing;
random_state=0 is needed to always have the same split)"""
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))


print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
# ----------------------------------------------------------------------------------------------
#                                       Checking training data for inadequacies:
# ----------------------------------------------------------------------------------------------

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

# ----------------------------------------------------------------------------------------------
#                                       Training k-mean k-neighbours model (k=1):
# ----------------------------------------------------------------------------------------------
"""
KNeighborsClassifier is a general class. Here we are creating its instance called knn.
"""
knn = KNeighborsClassifier(n_neighbors=1)  # using the closest neighbour for simplicity to create a class instance "knn"
knn.fit(X_train, y_train)  # fitting the 1-nearest neighbour model

# now let's see, how it works.
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# ----------------------------------------------------------------------------------------------
#                                       Testing k-neighbours model (k=1):
# ----------------------------------------------------------------------------------------------
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))


"""
1. y_pred == y_test returns an array, for which an element is either 0 or 1 for incorrect and correct guesses 
respectively;
2. We just take the mean of the elements in the array
"""
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#does pretty much the the same
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# ----------------------------------------------------------------------------------------------
#                                       Code summary:
# ----------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
