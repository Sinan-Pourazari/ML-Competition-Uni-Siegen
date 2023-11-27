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
    randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='entropy', min_samples_split=10)
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
#features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing
#doesnt do anything at the moment anyways

#selectorVariance= VarianceThreshold()
#features = selectorVariance.fit_transform(features)

'''
#try drop diferent features and see what nit does (brute force style)
drop_features =['feature_1','feature_3','feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10'
           , 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19'
           , 'feature_20', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29'
           , 'feature_30']
for i in drop_features:
    features=features.drop([i],axis=1)
    featurestemp = features.to_numpy()

    print('dropped feature: ', i ,'score: ',Run(featurestemp, labels))
'''
print(Run(features,labels))