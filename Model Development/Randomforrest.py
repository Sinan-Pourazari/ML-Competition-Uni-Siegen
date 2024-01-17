from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from Custom_Methods import *

def Feature_Elminination_CV(model,dataX,datay):
    # RFECV
    selector = RFECV(model, step=1, cv=5, n_jobs=-1, verbose=True, min_features_to_select=20)
    selector = selector.fit(dataX, datay)
    sel= selector.get_support()
    print(sel)
    return sel

def Run(X,y):

    #build randomforrest
    randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='gini', min_samples_split=3, n_jobs=-1, verbose=False)


    # split the data
    randomforest.fit(features, labels.to_numpy().flatten())


    scores = stratified_cross_fold_validator_for_smote(features,labels,5,randomforest)
    print("%0.7f F1-Macro with a standard deviation of %0.7f" % (scores.mean(), scores.std()))
    print(scores)

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

features,labels = removeOutlier(features,labels)

if __name__ == '__main__':

    print(Run(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13', 'feature_25', 'feature_9', 'feature_6', 'feature_4', 'feature_14', 'feature_28']],labels))