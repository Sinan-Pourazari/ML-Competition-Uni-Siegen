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
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif,mutual_info_classif
def Run(X,y):


    # scale
    #scale(X)

    #define model parameters
    GBmodel= GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False,
                                        loss='log_loss', criterion='friedman_mse')

    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)
    scale(Xtrain)
    scale(Xtest)
    GBmodel.fit(Xtrain, ytrain)
    print(GBmodel.feature_importances_)
    result = permutation_importance(GBmodel, X, y, n_repeats=10, random_state = 0)
    print(result.importances_mean)
    print(result.importances_std)
    #make Prediction

    pred =GBmodel.predict(Xtest)
    dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_f20')
    score = f1_score(ytest, pred, average='macro')
    scores = cross_val_score(GBmodel, X, y, cv=10, scoring='f1_macro')
    print(scores)
    print("%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#resample to counter inbalance
print('1: ', labels.value_counts()[1])
print('0: ', labels.value_counts()[0])
'''merged = pd.merge(features, labels, on='Id')
minority = merged.loc[merged['label'] == 1]
majority = merged.loc[merged['label']==0]
print(minority)
#minority = resample(minority, n_samples=600, random_state=42)
majority = resample(majority, n_samples=800, random_state=42, replace= True)
merged = pd.concat([majority, minority])
labels =merged['label']
merged.drop(['label'],inplace=True, axis=1)
features=merged
print(features)
#drop the id
features = features.drop(['Id'], axis=1)
#labels = labels.drop(['Id'], axis=1)
'''
#drop features without information
#
features.drop(['Id'], axis=1, inplace=True)
labels.drop(['Id'], axis=1, inplace=True)



#convert to numpyarray
#features=features.to_numpy()
labels=labels.to_numpy().flatten()


#feature selection
feature_names = features.columns.to_numpy().flatten()
SKB=SelectKBest(mutual_info_classif, k=28)
SKB.feature_names_in_=feature_names
features = SKB.fit_transform(features,labels)

print(SelectKBest.get_feature_names_out(SKB))

print(Run(features,labels))