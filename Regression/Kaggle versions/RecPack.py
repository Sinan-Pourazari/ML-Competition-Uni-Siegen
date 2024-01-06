from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
import pandas as pd

train = pd.read_csv('train_features.csv')
label = pd.read_csv('train_label.csv')

proc = DataFramePreprocessor('item', 'user', 'timestamp', 'rating')
proc.add_filter(MinUsersPerItem(5, item_ix='item_ix', user_ix='user_id'))
proc.add_filter(MinItemsPerUser(5, item_ix='item_ix', user_ix='user_id'))

# Assuming you have loaded a dataframe called df
interaction_matrix = proc.process(label)