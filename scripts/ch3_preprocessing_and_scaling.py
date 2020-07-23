"""
This program shows example of data preprocessing and scaling.
"""

# --------------------------------------------------------------------------------------------------------------
#                                       Importing packages.
# --------------------------------------------------------------------------------------------------------------

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
from sklearn.datasets import load_breast_cancer  # data
from sklearn.svm import SVC  # model
from sklearn.model_selection import train_test_split  # preparing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# --------------------------------------------------------------------------------------------------------------

# Different ways to rescale and preprocess a dataset
mglearn.plots.plot_scaling()
plt.show()

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))

# --------------------------------------------------------------------------------------------------------------

"""
The StandardScaler
in scikit-learn ensures that for each feature the mean is 0 and the variance is 1,
bringing all features to the same magnitude. However, this scaling does not ensure
any particular minimum and maximum values for the features.
"""

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)  # scoring on the scaled test set
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))


# --------------------------------------------------------------------------------------------------------------

"""
The RobustScaler
works similarly to the StandardScaler in that it ensures statistical properties
for each feature that guarantee that they are on the same scale. However, the RobustScaler uses
the median and quartiles,1 instead of mean and variance. This makes the RobustScaler ignore data
points that are very different from the rest (like measurement errors). These odd data points 
are also called outliers, and can lead to trouble for other scaling techniques.
"""
# --------------------------------------------------------------------------------------------------------------

"""
The MinMaxScaler, on the other hand, shifts the data such that all features are exactly between 0 and 1.
For the two-dimensional dataset this means all of the data is contained within the rectangle created 
by the x-axis between 0 and 1 and the y-axis between 0 and 1.
"""

scaler = MinMaxScaler()

scaler.fit(X_train)

 # transform data
X_train_scaled = scaler.transform(X_train)
# print dataset properties before and after scaling
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
	X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
	X_train_scaled.max(axis=0)))


# To apply the SVM to the scaled data, we also need to transform the test set.
# This is again done by calling the transform method, this time on X_test:
# transform test data
X_test_scaled = scaler.transform(X_test)
# print test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))

# Note:
# MinMaxScaler (and all the other scalers) always applies exactly the same transformation
# to the training and the test set. This means the transform method always subtracts the training
# set minimum and divides by the training set range, which might be different from the minimum and
# range for the test set.


# --------------------------------------------------------------------------------------------------------------
"""
Normalizer does a very different kind of rescaling. It scales each data point such that the feature vector 
has a Euclidean length of 1. In other words, it projects a data point on the circle (or sphere, in the case 
of higher dimensions) with a radius of 1. This means every data point is scaled by a different number 
(by the inverse of its length). This normalization is often used when only the direction (or angle)
of the data matters, not the length of the feature vector.
"""

# --------------------------------------------------------------------------------------------------------------
#                                               Summary:
# --------------------------------------------------------------------------------------------------------------

scaler = MinMaxScaler()  # or any of: StandardScaler(), Normalizer(), RobustScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now we can use scaled data in the model.
