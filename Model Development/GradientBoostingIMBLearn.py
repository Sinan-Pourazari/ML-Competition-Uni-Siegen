from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTENC, ADASYN, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE, SMOTE
from sklearn.neighbors import LocalOutlierFactor
from Custom_Methods import *

def Run(Xtrain, ytrain):
    # define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=True,
                                         loss='log_loss', criterion='friedman_mse')


    #to preven dataleak each train fold is oversampled indepenantly
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 5, GBmodel)

    # make Prediction
    # print(scores)
    print('scores: ', scores, "%0.2f F1-Macro with a standard deviation of %0.2f" % (np.mean(scores), np.std(scores)))


# read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

# drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

# drop features without information
features.drop(['feature_2','feature_20', 'feature_12'], axis=1, inplace=True)

features, labels = removeOutlier(features, labels)
if __name__ == '__main__':
    Run(features, labels)
