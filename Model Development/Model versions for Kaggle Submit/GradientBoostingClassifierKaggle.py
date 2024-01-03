from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import scale
from joblib import dump, load
from sklearn.utils import resample, compute_sample_weight
from imblearn.under_sampling import ClusterCentroids,CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours,AllKNN, InstanceHardnessThreshold, NeighbourhoodCleaningRule
from imblearn.combine import SMOTEENN
def Run(Xtrain,ytrain,Xtest):


    # scale
    #scale(X)

    #define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=26, random_state=42,
                                         verbose=True,
                                         loss='log_loss', criterion='friedman_mse', min_samples_leaf=16,
                                         min_samples_split=11, )

    weights = compute_sample_weight(class_weight='balanced' ,y=ytrain)
    for i in range(len(weights)):
        if(weights[i]>1):
            weights[i] = weights[i]*2

    GBmodel.fit(Xtrain, ytrain, weights)



    scores = cross_val_score(GBmodel, Xtrain, ytrain, cv=5, scoring='f1_macro')
    print(scores)
    #make Prediction
    pred =GBmodel.predict(Xtest)
    #dump(GBmodel, 'GBmodel6_wo_f2_downsamp_IMBLearn_NeighbourhoodCleaningRule_Kaggle1')

    return pred

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test_features = pd.read_csv('test_features.csv')


#drop the id
features = features.drop(['Id', 'feature_2'], axis=1)
labels = labels.drop(['Id'], axis=1)
test_features.drop(['Id', 'feature_2'], axis=1, inplace=True)



print('1: ', labels.value_counts()[1])
print('0: ', labels.value_counts()[0])

#convert to numpyarray
#features=features.to_numpy()
labels=labels.to_numpy().flatten()
print(labels)
result = Run(features, labels, test_features)

#export as csv file
#loop that makes the id for the predicited valus
idarr = np.array([])
for i in range(len(result)):
   idarr= np.append(idarr, i)
#make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
return_value=pd.DataFrame({'Id': idarr, 'label': result})
return_value=return_value.astype(int)
print(return_value)
#save it as file
return_value.to_csv('GBC36.csv', columns=['Id', 'label'], index=False)