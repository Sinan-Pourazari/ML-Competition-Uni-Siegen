#Linear_SVC. finds a linear seperator which is used to classify the data into two classes
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier

def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

    # scale
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)

    #make model
    MLPC = MLPClassifier(random_state=211,max_iter=99999, learning_rate_init=0.0041, solver='sgd', learning_rate='adaptive')
    MLPC.fit(Xtrain,ytrain)
    pred = MLPC.predict(Xtest)
    # calculate f1-score
    score = f1_score(ytest, pred)
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

print(Run(features, labels))