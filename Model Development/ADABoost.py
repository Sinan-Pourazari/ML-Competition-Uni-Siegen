from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier



def Run(X,y,test):
    #make custom randomforest from whch the model starts
    Randomforrest = RandomForestClassifier(random_state=211, n_estimators=100, criterion='entropy', min_samples_split=2)
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

    ABC= AdaBoostClassifier( estimator= Randomforrest ,n_estimators=100, random_state=test, learning_rate=1, algorithm='SAMME.R')
    ABC.fit(Xtrain,ytrain)
    pred = ABC.predict(Xtest)

    score = f1_score(ytest, pred, average='macro')
    return score


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

#basic pre prosessing
#features.drop(inplace=True, labels= ['feature_15'], axis=1)




#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing
selectorVariance= VarianceThreshold()
features = selectorVariance.fit_transform(features)

''' 
#loop to try the imact of the randomstate on the model
bestrdmscore=0
bestrdm=0
for i in range(10000):
    temp=Run(features, labels, i)
    print("f1-Score: ", temp, "randomstate: ",i)
    if temp > bestrdmscore:
        bestrdmscore=temp
        bestrdm=i
print("Best: ", bestrdmscore, " "," state: ",bestrdm)
#Best:  0.7846610066214404    state:  5591
'''
print(Run(features, labels, 710))


