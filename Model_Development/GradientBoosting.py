from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
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
from sklearn.utils.class_weight import compute_sample_weight
from Custom_Methods import *
def Run(Xtrain,ytrain,Xtest,ytest):

    #define model parameters
    GBmodel= GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=True,
                                         loss='log_loss', criterion='friedman_mse')

    weights = compute_sample_weight(class_weight='balanced' ,y=ytrain)

    for i in range(len(weights)):
        if(weights[i]>1):
            weights[i] = weights[i]*2

    print(weights)
    GBmodel.fit(Xtrain, ytrain,weights)
    print(GBmodel.get_params())


    print(GBmodel.feature_importances_)
    result = permutation_importance(GBmodel, Xtrain, ytrain, n_repeats=10, random_state = 0)
    print(result.importances_mean)
    print(result.importances_std)
    #make Prediction

    pred = GBmodel.predict(Xtest)


    #dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_f20')
    score = f1_score(ytest, pred, average='macro')
    temp = pd.DataFrame({'Test': ytest, 'Pred': pred})
    temp.to_csv('test.csv', columns=['Test', 'Pred'], index=False)

    tn, fp, fn, tp = confusion_matrix(ytest, pred).ravel()
    print('tn: ', tn, 'tp: ', tp, 'fn: ', fn, 'fp: ', fp)

    scores = cross_fold_validator(Xtrain, ytrain, 10, GBmodel)
    print(scores)
    print("%0.7f F1-Macro with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
    return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

features.drop(['Id','feature_2'], axis=1, inplace=True)
labels.drop(['Id'], axis=1, inplace=True)

#feature selection
'''feature_names = features.columns.to_numpy().flatten()
SKB=SelectKBest(mutual_info_classif, k='all')
SKB.feature_names_in_=feature_names
features = SKB.fit_transform(features,labels)

print(SelectKBest.get_feature_names_out(SKB))'''


#train test split before resampling to get "pure" test data
Xtrain, Xtest, ytrain, ytest = train_test_split(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13', 'feature_25', 'feature_9']], labels, test_size=0.1, random_state=26881)

'''#resample to counter inbalance
#merge labels to the corosponding sampels
merged = pd.merge(Xtrain, ytrain, on='Id')

#split into minority and majority for resamping porsesss
minority = merged.loc[merged['label'] == 1]
majority = merged.loc[merged['label']==0]
#resampling
#minority = resample(minority, n_samples=471, random_state=42, replace= False)
majority = resample(majority, n_samples=900, random_state=42, replace= False)
#put both resampled dataframes into one
merged = pd.concat([majority, minority])

#shuffle the samples so the data input instnt 800 times label 0 and 600 times label 1
merged = merged.sample(frac=1, random_state=42)
#extract labes from the meged set into its own dataframe again
ytrain =merged['label']
#drop labels from the mreged set to make it into a set with features only
merged.drop(['label'],inplace=True, axis=1)
Xtrain=merged

print('1: ', ytrain.value_counts()[1])
print('0: ', ytrain.value_counts()[0])
print(ytrain)'''

print('1: ', ytest.value_counts()[1])
print('0: ', ytest.value_counts()[0])

#drop features without information

'''Xtrain.drop(['Id'], axis=1, inplace=True)
Xtest.drop(['Id'], axis=1, inplace=True)
ytest.drop(['Id'],axis=1,inplace=True)
ytrain.drop(['Id'], axis=1, inplace=True)'''
#labels.drop(['Id'], axis=1, inplace=True)



#convert to numpyarray
#Xtrain=Xtrain.to_numpy()
#ytrain=ytrain.to_numpy().flatten()
#ytest = ytest.to_numpy().flatten()
#Xtest = Xtest.to_numpy()


if __name__ == '__main__':
    print(Run(Xtrain,ytrain,Xtest,ytest))