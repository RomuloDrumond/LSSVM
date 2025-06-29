print("Running GridSearchCV with LSSVC example...")

from lssvm.LSSVC import LSSVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

# Generate a sample binary classification dataset
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Define a parameter grid for LSSVC
param_grid = {
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'kernel_params': [{}, {'sigma': 0.5}, {'sigma': 1.0}] # Example kernel_params
}

# Instantiate LSSVC
lssvc_estimator = LSSVC()

# Instantiate GridSearchCV
# Note: For kernel_params, GridSearchCV will try each dict in the list.
# If a kernel does not use certain params (e.g. 'linear' ignores 'sigma'),
# those combinations will still be tried but LSSVC should ignore irrelevant kernel_params.
grid_search = GridSearchCV(
    estimator=lssvc_estimator,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy'
)

# Fit GridSearchCV on the generated data
grid_search.fit(X, y)

# Print the best parameters found
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Print the best score found
print("Best score found by GridSearchCV:")
print(grid_search.best_score_)

# Example of how to access the best estimator
best_lssvc = grid_search.best_estimator_
print("Best LSSVC estimator:", best_lssvc)

# You can also test the score method directly if needed
# score = best_lssvc.score(X, y)
# print("Score of the best estimator on the training data:", score)
