from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV

def Feature_Elminination_CV(model,dataX,datay):
    # RFECV
    selector = RFECV(model, step=1, cv=5, n_jobs=-1, verbose=True, min_features_to_select=20)
    selector = selector.fit(dataX, datay)
    sel= selector.get_support()
    print(sel)
    return sel

def Run(X,y):


    # scale
    #Xtrain = scale(Xtrain)
    #Xtest = scale(Xtest)

    #build randomforrest
    randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='gini', min_samples_split=3, n_jobs=-1, verbose=False)


    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    randomforest.fit(Xtrain, ytrain)


    #make Prediction
    pred =randomforest.predict(Xtest)

    score = f1_score(ytest, pred, average='macro')
    scores = cross_val_score(randomforest, Xtrain, ytrain, cv=10, scoring='f1_macro')
    print("%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print(scores)
    return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
'''sel= [False,  True, False, False, False, False, False, False, False,  True, False,  True,
 False, False, False, False,  True,  True,  True,  True, False, False, False, False,
  True, False, False, False, False,  True, False]'''
#drop features without information
#features.drop(sel, axis=1, inplace=True)
#features = features.loc[:, sel]
#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#PCA

'''
pca = PCA(n_components=2)
pca.fit_transform(features)
print(features)'''
#basic preprossesing
#doesnt do anything at the moment anyways

#selectorVariance= VarianceThreshold()
#features = selectorVariance.fit_transform(features)

print(Run(features,labels))