from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import cross_val_score



def Run(X,y,test):

    #make custom randomforest from whch the model starts
    RDMForest = RandomForestClassifier(random_state=211, n_estimators=5, criterion='log_loss', min_samples_split=10, n_jobs=-1,
                                       min_impurity_decrease= 0.0, verbose=True)
    GBmodel= GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=True)

    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=211)

    # Kfold crossvalidation
    kf = KFold(n_splits=5)
    kf.get_n_splits()

    ABC= AdaBoostClassifier( estimator= GBmodel ,n_estimators=50, random_state=211, learning_rate=1, algorithm='SAMME.R')
    ABC.fit(Xtrain,ytrain)
    pred = ABC.predict(Xtest)
    #cross vaidation
    scores = cross_val_score(ABC, Xtrain, ytrain, cv=10, scoring='f1_macro')
    print("%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print(scores)

    score = f1_score(ytest, pred, average='macro')
    return score


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

#Garbage feature removal
features.drop(inplace=True, labels= ['feature_2'], axis=1)




#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing
selectorVariance= VarianceThreshold()
features = selectorVariance.fit_transform(features)

''' 

'''
print(Run(features, labels, 710))


