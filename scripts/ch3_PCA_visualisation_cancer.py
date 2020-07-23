"""
This program shows example of using PCA.
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
from sklearn.decomposition import PCA
from sklearn.svm import SVC  # model
from sklearn.model_selection import train_test_split  # preparing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------------------------------
#                                       Illustration:
# --------------------------------------------------------------------------------------------------------------


# The following example (Figure 3-3) illustrates the effect of PCA on a synthetic two-dimensional data set:
mglearn.plots.plot_pca_illustration()

# --------------------------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# --------------------------------------------------------------------------------------------------------------

cancer = load_breast_cancer()

# --------------------------------------------------------------------------------------------------------------
#                                       Pre-processing and scaling:
# --------------------------------------------------------------------------------------------------------------

# Scaling the data, so features have mean=0 and variance=1
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to breast cancer data
pca.fit(X_scaled)

# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# plot first vs. second principal component, colored by class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# plt.show()


"""
The principal components correspond to directions in the original data, so they are 
combinations of the original features. However, these combinations are usually very complex, 
as we’ll see shortly. The principal components themselves are stored in the components_ attribute 
of the PCA object during fitting:
"""
print("PCA component shape: {}".format(pca.components_.shape))


"""
Each row in components_ corresponds to one principal component, and they are sorted 
by their importance (the first principal component comes first, etc.). 
The columns correspond to the original features attribute of the PCA.
"""
print("PCA components:\n{}".format(pca.components_))

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")

plt.show()


# --------------------------------------------------------------------------------------------------------------
#                                               Summary:
# --------------------------------------------------------------------------------------------------------------

"""

"""