"""
This program shows example of using pipelines.
"""

# ----------------------------------------------------------------------------------------------
#                                       Importing packages.
# ----------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, make_pipeline

# ----------------------------------------------------------------------------------------------
#                          Importing the data:
# ----------------------------------------------------------------------------------------------

# load and split the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# ----------------------------------------------------------------------------------------------
#                           Scaling the data:
# ----------------------------------------------------------------------------------------------

# compute minimum and maximum on the training data
scaler = MinMaxScaler().fit(X_train)
# rescale the training data
X_train_scaled = scaler.transform(X_train)

# ----------------------------------------------------------------------------------------------
#                           Fitting the model:
# ----------------------------------------------------------------------------------------------
svm = SVC()
# learn an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# ----------------------------------------------------------------------------------------------
#                           Model assessment on the test set:
# ----------------------------------------------------------------------------------------------

# scale the test data and score the scaled data
X_test_scaled = scaler.transform(X_test)
print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# Test score: 0.95

################################################################################################
# ----------------------------------------------------------------------------------------------
#                           Parameter Selection with Preprocessing:
# ----------------------------------------------------------------------------------------------
################################################################################################


# ----------------------------------------------------------------------------------------------
#                             A naive INCORRECT approach
# ----------------------------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

# for illustration purposes only, don't use this code!
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best set score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("Best parameters: ", grid.best_params_)
# Best cross-validation accuracy: 0.98
# Best set score: 0.97
# Best parameters:  {'C': 1, 'gamma': 1}


# ----------------------------------------------------------------------------------------------
#                       One of the correct approaches using pipelines
# ----------------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline

pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))
# Test score: 0.95 - same as in the initial example

# ----------------------------------------------------------------------------------------------
#                   		Pipelines for grid searches
# ----------------------------------------------------------------------------------------------
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# dictionary keys contain of pipeline component name 'svm', double underscore '__', and parameter name 'C' and 'gamma'


grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))
# Best cross-validation accuracy: 0.98
# Test set score: 0.97
# Best parameters: {'svm__C': 1, 'svm__gamma': 1}

# ----------------------------------------------------------------------------------------------
#                           Illustrating Information Leakage
# ----------------------------------------------------------------------------------------------
"""
Let’s consider a synthetic regression task with 100 samples and 1,000 features that are sampled independently from a
Gaussian distribution. We also sample the response from a Gaussian distribution. Given the way we created the dataset,
there is no relation between the data, X, and the target, y (they are independent), so it should not be possible to
learn anything from this dataset.

We will now do the following. First, select the most informative of the 10 features using SelectPercentile feature
selection, and then we evaluate a Ridge regressor using cross-validation:
"""

# creating the data
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))

# selecting the most important features (500 features <=> 5 percent out of 10000)
from sklearn.feature_selection import SelectPercentile, f_regression
select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
print("X_selected.shape: {}".format(X_selected.shape))


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
print("Cross-validation accuracy (cv only on ridge): {:.2f}".format(
    np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))))
# X_selected.shape: (100, 500)
# Cross-validation accuracy (cv only on ridge): 0.91

"""
The mean R^2 computed by cross-validation is 0.91, indicating a very good model. This clearly cannot be right, as our
data is entirely random. What happened here is that our feature selection picked out some features among the 10,000
random features that are (by chance) very well correlated with the target. Because we fit the feature selection outside
of the cross-validation, it could find features that are correlated both on the training and the test folds. The
information we leaked from the test folds was very informative, leading to highly unrealistic results. Let’s compare
this to a proper cross-validation using a pipeline:
"""

pipe = Pipeline([("select", SelectPercentile(score_func=f_regression,
                                             percentile=5)),
                 ("ridge", Ridge())])
print("Cross-validation accuracy (pipeline): {:.2f}".format(
    np.mean(cross_val_score(pipe, X, y, cv=5))))
# Cross-validation accuracy (pipeline): -0.25
# As expected - we get very poor cross-validation accuracy (negative R^2 means we predict worse than mean prediction)

"""
This time, we get a negative R^2 score, indicating a very poor model. Using the pipeline, the feature selection is now
inside the cross-validation loop. This means features can only be selected using the training folds of the data, not the
test fold. The feature selection finds features that are correlated with the target on the training set, but because the
data is entirely random, these features are not correlated with the target on the test set. In this example, rectifying 
the data leakage issue in the feature selection makes the difference between concluding that a model works very well and 
concluding that a model works not at all.
"""
# ----------------------------------------------------------------------------------------------
#                           The General Pipeline Interface
# ----------------------------------------------------------------------------------------------
"""
The only requirement for estimators in a pipeline is that all but the last step need to have a transform method, so they 
can produce a new representation of the data that can be used in the next step.
"""

# ----------------------------------------------------------------------------------------------
#                           Convenient Pipeline Creation with make_pipeline
# ----------------------------------------------------------------------------------------------
"""
Creating a pipeline using the syntax described earlier is sometimes a bit cumbersome, and we often don’t need 
user-specified names for each step. There is a convenience function, make_pipeline, that will create a pipeline for us
and automatically name each step based on its class. The syntax for make_pipeline is as follows.
"""

from sklearn.pipeline import make_pipeline
# standard syntax
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# abbreviated syntax
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

# the steps of pipe_short are automatically named:
print("Pipeline steps:\n{}".format(pipe_short.steps))
# The steps are named minmaxscaler and svc

# ----------------------------------------------------------------------------------------------
#                                   Accessing Step Attributes
# ----------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
# fit the pipeline defined before to the cancer dataset
pipe.fit(cancer.data)
# extract the first two principal components from the "pca" step
components = pipe.named_steps["pca"].components_
print("components.shape: {}".format(components.shape))


# ----------------------------------------------------------------------------------------------
#                     Accessing Attributes in a Grid-Searched Pipeline
# ----------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())

"""
Next, we create a parameter grid. As explained in Chapter 2, the regularization parameter to tune for LogisticRegression
is the parameter C. We use a logarithmic grid for this parameter, searching between 0.01 and 100. Because we used the 
ake_pipeline function, the name of the LogisticRegression step in the pipeline is the lowercased class name,
logisticregression. To tune the parameter C, we there‐ fore have to specify a parameter grid for logisticregression__C:
"""

param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=4)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

# best estimator
print("Best estimator:\n{}".format(grid.best_estimator_))

# best version of logistic regression
print("Logistic regression step:\n{}".format(grid.best_estimator_.named_steps["logisticregression"]))

# accessing the coefficients
print("Logistic regression coefficients:\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))


# ----------------------------------------------------------------------------------------------
#                   Grid-Searching Preprocessing Steps and Model Parameters
# ----------------------------------------------------------------------------------------------

from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(),
        Ridge())

param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1),
                vmin=0, cmap="viridis")
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),
           param_grid['polynomialfeatures__degree'])
plt.colorbar()
plt.show()


print("Best parameters: {}".format(grid.best_params_))
# Best parameters: {'polynomialfeatures__degree': 2, 'ridge__alpha': 10}
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
# Test-set score: 0.77


# Let’s run a grid search without polynomial features for comparison:
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Score without poly features: {:.2f}".format(grid.score(X_test, y_test)))
# Score without poly features: 0.63

"""
As we would expect looking at the grid search results visualized in Figure 6-4, using no polynomial features leads to
decidedly worse results.
"""

# ----------------------------------------------------------------------------------------------
#                           Grid-Searching Which Model To Use
# ----------------------------------------------------------------------------------------------

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

from sklearn.ensemble import RandomForestClassifier
param_grid = [
    {'classifier': [SVC()],
     'preprocessing': [StandardScaler(), None],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]


X_train, X_test, y_train, y_test = train_test_split(
     cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

"""
Best params:
{'classifier': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False),
    'classifier__C': 10,
    'classifier__gamma': 0.01,
    'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True)}

Best cross-validation score: 0.99

Test-set score: 0.98
"""
