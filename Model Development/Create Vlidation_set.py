import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from lightgbm import LGBMClassifier



# Load the data
X = pd.read_csv('train_features.csv')
y = pd.read_csv('train_label.csv')

temp= pd.merge(X,y, on='Id')
# drop the id so it isnt used as a training feature
Train, Validation = train_test_split(temp, random_state=42, test_size=0.1)

# drop some more stuff
Validation.to
