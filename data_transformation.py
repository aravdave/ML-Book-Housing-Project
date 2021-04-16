# %%
from numpy.core.numeric import full
from data_exploration import get_strat_data
import pandas as pd
# %%
strat_train_set, strat_test_set = get_strat_data()
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
# %%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)
# %%
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
# %%
housing_cat = housing[['ocean_proximity']]
housing_cat.value_counts()
# %%
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# %%
ordinal_encoder.categories_
""" Notice here that the categories are randomly assigned a number,
    which causes a problem since two similar variables could be on
    opposite ends of the number scale """
# %%
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
# %%
""" It's very useful that housing_cat_1hot is a sparse matrix and
    not a Numpy arrray because a sparse matrix doesn't waste memory
    storing the location of thousands of 0s like Numpy Arrays.
    Instead, sparse matrix efficiently only stores the location of
    nonzero elements.
    """
# %%
cat_encoder.categories_
# %%
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
# %%
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
# %%
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# %%
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
    
# %%
# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# %%
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:', [round(elem, 2) for elem in lin_reg.predict(some_data_prepared)])
print('Labels:', list(some_labels))
# %%
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# %%
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
# We use a neg mean squared error because cross_value_score expects a uti
# -lity function. Greater is better.
tree_rmse_scores = np.sqrt(-scores)
# %%
def display_scores(scores):
    print('Scores:', scores)
    print('Mean', scores.mean())
    print("Standard Deviation", scores.std())
# %%
display_scores(tree_rmse_scores)
# %%
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring='neg_mean_squared_error', cv=10)
lin_rmse = np.sqrt(-lin_scores)
display_scores(lin_rmse)
# %%
import joblib

joblib.dump(lin_reg, 'lin_reg.pkl')
joblib.dump(tree_reg, 'tree_reg.pkl')
# lin_reg_loaded = joblib.load('lin_reg.pkl)
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3,10,30], 'max_features' : [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)

#%%
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVR

# param_grid = [
#     {'kernel': ['linear'], 'C': [2,4,16,(16**2)]},
#     {'kernel': ['rbf'], 'C': [2,4,16,(16**2)], 'gamma': [2,4,16,(16**2)]}
# ]

# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True, verbose=2)
# grid_search.fit(housing_prepared, housing_labels)
#%%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal

param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distribs,
                            n_iter=50, cv=5, scoring='neg_mean_squared_error',
                            verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# %%
negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
print(rmse)

# %%
print(grid_search.best_params_)
print(grid_search.best_estimator_)
# %%
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score), params)
# %%
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
# %%
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_hhold']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# %%
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# %%
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions-y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
        scale = stats.sem(squared_errors)))
# %%
k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
print(top_k_feature_indices)
print(np.array(attributes)[top_k_feature_indices])

print(sorted(zip(feature_importances, attributes), reverse=True)[:k])

#%%
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
print(housing_prepared_top_k_features[0:3])
print(housing_prepared[0:3, top_k_feature_indices])
# %%
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])
# %%
prepare_select_and_predict_pipeline.fit(housing, housing_labels)
# %%
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print('Predictions:\t', prepare_select_and_predict_pipeline.predict(some_data))
print('Labels:\t\t', list(some_labels))
# %%
param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

from sklearn.model_selection import GridSearchCV
grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)
# %%
grid_search_prep.best_params_
# {'feature_selection__k': 15,
# 'preparation__num__imputer__strategy': 'most_frequent'}
# %%
