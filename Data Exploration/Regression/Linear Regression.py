import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import scale


def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

    # scale
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)
    reg = LinearRegression()
    reg.fit(Xtrain,ytrain)
    pred = reg.predict(Xtest)
    score = np.sqrt(mean_squared_error(ytest, pred))
    return score
#read csv
features = pd.read_csv('Kaggle versions/train_features.csv')
labels = pd.read_csv('Kaggle versions/train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()
#basic preprossesing
#doesnt do anything at the moment anyways

#selectorVariance= VarianceThreshold()
#features = selectorVariance.fit_transform(features)

print(Run(features,labels))