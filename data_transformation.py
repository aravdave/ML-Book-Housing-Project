# %%
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
from sklearn.base import BaseEncoder, TransformerMixin