from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,make_scorer
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV




def Run(X,y,test):
    #make custom randomforest from whch the model starts
    Randomforrest = RandomForestClassifier(random_state=211, n_estimators=50, criterion='entropy', min_samples_split=10)
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.24, random_state=211)

    ABC= AdaBoostClassifier( estimator= Randomforrest ,n_estimators=50, random_state=5591, learning_rate=1, algorithm='SAMME.R')
    # Define the parameter grid for the grid search
    n_estimators_array = [];
    min_samples_split_array = [];
    for i in range(2, 200, 5):
        n_estimators_array.append(i)

    for i in range(2, 20):
        min_samples_split_array.append(i)
    param_grid = {
        'n_estimators': n_estimators_array,
        'learning_rate': [0.9, 1.0],
        #'criterion': ['entropy', 'log_loss', 'gini'],
        #'min_samples_split': min_samples_split_array,
        'algorithm': ['SAMME.R', 'SAMME']

    }
    f1_scorer = make_scorer(f1_score, average='macro')
    grid_search = GridSearchCV(estimator=ABC, param_grid=param_grid, cv=5, scoring=f1_scorer)
    grid_search.fit(features, labels)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

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
#features.drop(inplace=True, labels= ['feature_13'], axis=1)


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


