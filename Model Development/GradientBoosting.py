from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import scale
from joblib import dump, load
from sklearn.utils import resample

def Run(X,y):


    # scale
    #scale(X)

    #define model parameters
    GBmodel= GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=True,
                                        loss='log_loss', criterion='friedman_mse')


    # split the data
    ''' Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    scale(Xtrain)
    scale(Xtest)'''
    GBmodel.fit(X, y)
    print(GBmodel.feature_importances_)


    #make Prediction
    #pred =GBmodel.predict(Xtest)
    dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel5_wo_f2_downsamp_upsamp')
   # score = f1_score(ytest, pred, average='macro')
    scores = cross_val_score(GBmodel, X, y, cv=10, scoring='f1_macro')
    print(scores)
    print("%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    #return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#resample to counter inbalance
merged = pd.merge(features, labels, on='Id')
minority = merged.loc[merged['label'] == 1]
majority = merged.loc[merged['label']==0]
minority = resample(minority, n_samples=700, random_state=42)
majority = resample(majority, n_samples=800, random_state=42)
merged = pd.concat([majority, minority])
labels =merged['label']
merged.drop(['label'],inplace=True, axis=1)
features=merged

#drop the id
features = features.drop(['Id'], axis=1)
#labels = labels.drop(['Id'], axis=1)

#drop features without information
#
features.drop(['feature_2'], axis=1, inplace=True)


#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()



print(Run(features,labels))