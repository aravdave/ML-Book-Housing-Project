# %%
import os, tarfile, urllib
DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# %%
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# %%
# fetch_housing_data()
# %%
import pandas as pd
# %%
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
# %%
housing = load_housing_data()
housing.head()
# %%
housing.info()
# %%
housing['ocean_proximity'].value_counts()
# %%
housing.describe()
# %%
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
# %%
import numpy as np
# %%
## Create your own test/train dataframes (why would you waste time doing this?)
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# %%
# train_set, test_set = split_train_test(housing, .2)
# print(len(train_set))
# print(len(test_set))
# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=.2, random_state=42)
# %%
housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
                                labels=[1,2,3,4,5])
# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# %%
strat_test_set.income_cat.value_counts() / len(strat_test_set)
# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# %%
def get_strat_data():
    return strat_train_set, strat_test_set
# # %%
# housing = strat_train_set.copy()

# # %%
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.1)
# # %%
# housing.plot(kind='scatter', x="longitude", y="latitude", alpha=.4, s=housing.population/100,
#                 label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'),
#                 colorbar=True,
# )
# plt.legend()
# # %%
# corr_matrix=housing.corr()
# corr_matrix.median_house_value.sort_values(ascending=False)
# # %%
# from pandas.plotting import scatter_matrix

# attributes=['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
# scatter_matrix(housing[attributes], figsize=(12,8))
# # %%
# housing.plot(kind='scatter', x= 'median_income', y='median_house_value', alpha=.1)
# # %%
# housing['rooms_per_household'] = housing['total_rooms']/housing['households']
# housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# housing['population_per_household'] = housing['population']/housing['households']
# # %%
# corr_matrix = housing.corr()
# corr_matrix['median_house_value'].sort_values(ascending=False)