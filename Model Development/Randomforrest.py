from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier



def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

    # scale
    #Xtrain = scale(Xtrain)
    #Xtest = scale(Xtest)

    #build randomforrest
    randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='entropy', min_samples_split=2)
    randomforest.fit(Xtrain, ytrain)

    #make Prediction
    pred =randomforest.predict(Xtest)

    score = f1_score(ytest, pred, average='macro')
    return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
features.drop(['feature_0', 'feature_21'], axis=1, inplace=True)




#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing

selectorVariance= VarianceThreshold()
features = selectorVariance.fit_transform(features)

print(Run(features, labels))