from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder
def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

    # scale
    #Xtrain = scale(Xtrain)
    #Xtest = scale(Xtest)

    #make model
    MLP = MLPRegressor(random_state=211,max_iter=1000)




    MLP.fit(Xtrain,ytrain)
    pred = MLP.predict(Xtest)
    # calculate f1-score
    score =np.sqrt(mean_squared_error(ytest, pred))
    return score


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)

labels = labels.drop(['Id'], axis=1)

#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing
enc = OrdinalEncoder()

#for what ever reason this decraees the performance by arround 2.6%
selectorVariance= VarianceThreshold()
features = selectorVariance.fit_transform(features)

print("RMSE: ", Run(features, labels))