"""
This code gives an example of using madel based automatic variable selection.
"""

###############################################################################
# -----------------------------------------------------------------------------
#                           Importing packages:
# -----------------------------------------------------------------------------
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data
from sklearn.datasets import load_breast_cancer
import mglearn

# preparing data
from sklearn.model_selection import train_test_split

# Representing Data and Engineering Features: Automatic Feature Selection
# univariate statistics
from sklearn.feature_selection import SelectPercentile
# model-based feature selection
from sklearn.feature_selection import SelectFromModel
# recursive feature elimination
from sklearn.feature_selection import RFE

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# -----------------------------------------------------------------------------
#                           Getting data:
# -----------------------------------------------------------------------------
###############################################################################

cancer = load_breast_cancer()

# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])

###############################################################################
# -----------------------------------------------------------------------------
#                           Manipulating the data:
# -----------------------------------------------------------------------------
###############################################################################

X_train, X_test, y_train, y_test = train_test_split(
	X_w_noise, cancer.target, random_state=0, test_size=.5)

###############################################################################
# -----------------------------------------------------------------------------
#           Selecting the variables (f-tests without the model):
# -----------------------------------------------------------------------------
###############################################################################

# use f_classif (the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

mask = select.get_support()
print(mask)
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

plt.show()


###############################################################################
# -----------------------------------------------------------------------------
#                       Fitting logistic regression:
# -----------------------------------------------------------------------------
###############################################################################

# transform test data
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)

print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
	lr.score(X_test_selected, y_test)))

"""
This was a very simple synthetic example, and out‐ comes on real data 
are usually mixed. Univariate feature selection can still be very helpful, 
though, if there is such a large number of features that building a model 
on them is infeasible, or if you suspect that many features are completely 
uninformative.
"""

###############################################################################
# -----------------------------------------------------------------------------
#                   Model-based feature selection:
# -----------------------------------------------------------------------------
###############################################################################

select = SelectFromModel(
	RandomForestClassifier(n_estimators=100, random_state=42),
	threshold="median")

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

mask = select.get_support()
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

plt.show()

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Test score: {:.3f}".format(score))

###############################################################################
# -----------------------------------------------------------------------------
#                   Iterative Feature Selection:
# -----------------------------------------------------------------------------
###############################################################################

# recursive feature elimination
select = RFE(RandomForestClassifier(
	n_estimators=100, random_state=42
	), n_features_to_select=40)

select.fit(X_train, y_train)
# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")


X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Test score: {:.3f}".format(score))

print("Test score: {:.3f}".format(select.score(X_test, y_test)))

###############################################################################
# -----------------------------------------------------------------------------
#                   Utilizing Expert Knowledge:
# -----------------------------------------------------------------------------
###############################################################################

citibike = mglearn.datasets.load_citibike()

plt.figure(figsize=(10, 3))
xticks = pd.date_range(
	start=citibike.index.min(), end=citibike.index.max(), freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Rentals")
