from imblearn.over_sampling import SMOTE
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
from sklearn.inspection import permutation_importance
def Run(Xtrain,ytrain,Xtest,ytest, feature_names):

    #define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=False,
                                         loss='log_loss', criterion='friedman_mse')
    GBmodel.fit(Xtrain,ytrain)

    per_imp = permutation_importance(GBmodel, Xtest, ytest,
    n_repeats = 20000,random_state = 0, n_jobs=-1, )
    print('#### PERMUTATION IMPORTANCE ####')
    for i in per_imp.importances_mean.argsort()[::-1]:

            print(f"{feature_names[i]:} "
            f"{per_imp.importances_mean[i]:.3f}"
            f" +/- {per_imp.importances_std[i]:.3f}")

    print('#### FEATURE IMPORTANCE ####')
    print(GBmodel.feature_importances_)
    #make Prediction

    pred = GBmodel.predict(Xtest)


    #dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_f20')
    score = f1_score(ytest, pred, average='macro')
    temp = pd.DataFrame({'Test': ytest, 'Pred': pred})
    temp.to_csv('test.csv', columns=['Test', 'Pred'], index=False)

    tn, fp, fn, tp = confusion_matrix(ytest, pred).ravel()
    print('tn: ', tn, 'tp: ', tp, 'fn: ', fn, 'fp: ', fp)

    scores = cross_val_score(GBmodel, Xtrain, ytrain, cv=5, scoring='f1_macro', n_jobs=-1)
    print(scores)
    print("%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

features.drop(['Id'], axis=1, inplace=True)
labels.drop(['Id'], axis=1, inplace=True)




#train test split before resampling to get "pure" test data
features, Xtest,labels, ytest = train_test_split(features, labels, test_size=0.2, random_state=26881)


#drop features without information
#labels.drop(['Id'], axis=1, inplace=True)
smo = SMOTE(random_state=42, n_jobs=-1)
X_res, y_res = smo.fit_resample(features, labels)
features =X_res
labels = y_res
labels = pd.DataFrame(labels)


#convert to numpyarray
Xtrain=features.to_numpy()
ytrain=labels.to_numpy().flatten()
ytest = ytest.to_numpy().flatten()
Xtest = Xtest.to_numpy()


print(Run(Xtrain,ytrain,Xtest,ytest, features.columns))