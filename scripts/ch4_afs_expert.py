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
import mglearn

# preparing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

# Representing Data and Engineering Features: Automatic Feature Selection
# univariate statistics
from sklearn.feature_selection import SelectPercentile
# model-based feature selection
from sklearn.feature_selection import SelectFromModel
# recursive feature elimination
from sklearn.feature_selection import RFE

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

###############################################################################
# -----------------------------------------------------------------------------
#               Importing and getting to know data:
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

###############################################################################
# -----------------------------------------------------------------------------
#                           Data manipulation:
# -----------------------------------------------------------------------------
###############################################################################

# extract the target values (number of rentals)
y = citibike.values
# convert the time to POSIX time using "%s"
# X = citibike.index.strftime("%s").astype("int").reshape(-1, 1) - does not
# work anymore
X = citibike.index.astype("int64").values.reshape(-1, 1) // 10**9


# use the first 184 data points for training, and the rest for testing
n_train = 184


# function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
	# split the given features into a training and a test set
	X_train, X_test = features[:n_train], features[n_train:]
	# also split the target array
	y_train, y_test = target[:n_train], target[n_train:]

	# fitting the model:
	regressor.fit(X_train, y_train)
	print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))

	# plotting the graph
	y_pred = regressor.predict(X_test)
	y_pred_train = regressor.predict(X_train)
	plt.figure(figsize=(10, 3))
	plt.xticks(
		range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,
		ha="left")
	plt.plot(range(n_train), y_train, label="train")
	plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
	plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
	plt.plot(
		range(n_train, len(y_test) + n_train), y_pred, '--',
		label="prediction test")
	plt.legend(loc=(1.01, 0))
	plt.xlabel("Date")
	plt.ylabel("Rentals")


###############################################################################
# -----------------------------------------------------------------------------
#                           Fitting models:
# -----------------------------------------------------------------------------
###############################################################################

# -----------------------------------------------------------------------------
#                           Random forest:
# -----------------------------------------------------------------------------

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
plt.figure()
eval_on_features(X, y, regressor)

"""
Decision tree based models can not extrapolate, so they give the closest 
point in the training set—which is the last time it observed any data.
"""

# using hour as predictor
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

# using hour and day of week as predictors
X_hour_week = np.hstack(
	[citibike.index.dayofweek.values.reshape(-1, 1),
		citibike.index.hour.values.reshape(-1, 1)]
	)
eval_on_features(X_hour_week, y, regressor)


# -----------------------------------------------------------------------------
#                           Linear regression:
# -----------------------------------------------------------------------------
"""
Linear regression with continuous time scales for hours and days
of week
"""
eval_on_features(X_hour_week, y, LinearRegression())

# -----------------------------------------------------------------------------
"""
Ridge regression with hours and days of the week represented as categories
"""
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

eval_on_features(X_hour_week_onehot, y, Ridge())


# -----------------------------------------------------------------------------
"""
Ridge regression with:
1) hours and days of the week represented as categories
2) their interactions (only, polynomials of degrees 0 and 1 are not produced
as well as interactions of variables with itself) 
"""
poly_transformer = PolynomialFeatures(
	degree=2, interaction_only=True,
	include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)


###############################################################################
# -----------------------------------------------------------------------------
#               Getting the coefficients for linear model:
# -----------------------------------------------------------------------------
###############################################################################

hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features = day + hour

features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature magnitude")
plt.ylabel("Feature")





plt.show()



