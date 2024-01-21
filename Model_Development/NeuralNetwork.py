from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold

def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

    # scale
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)

    #make model
    MLPC = MLPClassifier(random_state=211,max_iter=99999, learning_rate_init=0.00419,
                         solver='sgd', learning_rate='adaptive', activation='relu',
                         shuffle= True, n_iter_no_change=3, momentum= 0.9, nesterovs_momentum= True,
                         hidden_layer_sizes=(100,), tol= 1e-4, verbose=True)

    #MLPC = MLPClassifier(random_state=211, max_iter=10000)


    MLPC.fit(Xtrain,ytrain)
    pred = MLPC.predict(Xtest)
    # calculate f1-score
    score = f1_score(ytest, pred, average='macro')
    return score


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id','feature_2' ,'feature_21'], axis=1)
labels = labels.drop(['Id'], axis=1)


#0.7591787866128608
#7707335759433058 f9
print(features)


#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing

#for what ever reason this decraees the performance by arround 2.6%
#selectorVariance= VarianceThreshold()
#features = selectorVariance.fit_transform(features)

print(Run(features, labels))