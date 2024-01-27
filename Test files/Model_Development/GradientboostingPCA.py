from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def Run(Xtrain,ytrain,Xtest,ytest):

    #define model parameters
    GBmodel= GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=False,
                                         loss='log_loss', criterion='friedman_mse')


    GBmodel.fit(Xtrain, ytrain)



    pred = GBmodel.predict(Xtest)


    #dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_f20')
    score = f1_score(ytest, pred, average='macro')
    '''temp = pd.DataFrame({'Test': ytest, 'Pred': pred})
    temp.to_csv('test.csv', columns=['Test', 'Pred'], index=False)'''
    tn, fp, fn, tp = confusion_matrix(ytest, pred).ravel()
    #print('tn: ', tn, 'tp: ', tp, 'fn: ', fn, 'fp: ', fp)

    scores = cross_val_score(GBmodel, Xtrain, ytrain, cv=5, scoring='f1_macro', n_jobs=-1)
    #print(scores)
    print(scores, "%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()), score)
    #return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#drop ID as well as feature 2 as it is useless
features.drop(['Id','feature_2'], axis=1, inplace=True)
labels.drop(['Id'], axis=1, inplace=True)

#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#sclas the data for pcascaler = StandardScaler()
#features_t = scaler.fit_transform(features)
Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1, random_state=54)

#pca
for i in range(1,len(Xtrain)):
    pca = PCA(random_state=42,n_components=i, whiten=True,copy=True)
    decomposed_train = pca.fit_transform(Xtrain)
    decomposed_test = pca.fit_transform(Xtest)
    print('components: ', i)
    concatenated_train = np.concatenate((Xtrain,decomposed_train), axis=1)
    concatenated_test = np.concatenate((Xtest,decomposed_test), axis=1)
    Run(concatenated_train, ytrain, concatenated_test, ytest)