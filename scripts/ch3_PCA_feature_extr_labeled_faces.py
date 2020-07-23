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

# data
from sklearn.datasets import fetch_lfw_people

# splitting for ML
from sklearn.model_selection import train_test_split  # preparing

# dimensionality reduction
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# models
from sklearn.svm import SVC  # model
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------------------------------------------------------------------
#                                       Illustration:
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
#                                       Importing and getting to know our data:
# --------------------------------------------------------------------------------------------------------------

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fix, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])


print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))


# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names
for i, (count, name) in enumerate(zip(counts, people.target_names)):
	print("{0:25} {1:3}".format(name, count), end=' ')
	if(i+1) % 3 == 0:
		print()

print()
# --------------------------------------------------------------------------------------------------------------
#                                       Pre-processing and scaling:
# --------------------------------------------------------------------------------------------------------------

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255.


# --------------------------------------------------------------------------------------------------------------
#                                   Learning with KNeighborsClassifier:
# --------------------------------------------------------------------------------------------------------------

"""
A common task in face recognition is to ask if a previously unseen face belongs 
to a known person from a database.

A simple solution is to use a one-nearest-neighbor classifier that looks for the most similar face image 
to the face you are classifying. This classifier could in principle work with only a single training 
example per class. Let’s take a look at how well KNeighborsClassifier does here:
"""

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("\nTest set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))

# --------------------------------------------------------------------------------------------------------------

"""
From now we will use whitening - scaling option of PCA, so the components have the dame range.
It has a similar to StandardScaler effect on data.
"""
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape: {}".format(X_train_pca.shape))

# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("\nTest set score of 1-nn: {:.2f}".format(knn.score(X_test_pca, y_test)))
# score improves


print("pca.components_.shape: {}".format(pca.components_.shape))


fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})

for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
	ax.imshow(component.reshape(image_shape), cmap='viridis')
	ax.set_title("{}. component".format((i + 1)))

mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

plt.show()