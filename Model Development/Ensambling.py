import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from Custom_Methods import *

def Run(Xtrain, ytrain):
    # define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=284, learning_rate=0.1, max_depth=None, random_state=211, min_samples_leaf=2,
                                         verbose=True, min_samples_split= 152, min_weight_fraction_leaf=0.00020527722339855038,
                                         loss='log_loss', criterion='friedman_mse')


    #order= permutation_tester(Xtrain,ytrain,GBmodel,verbose=True)
    #to preven dataleak each train fold is oversampled indepenantly
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 5, GBmodel)
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 5, GBmodel)

    print('scores smote: ', scores, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores), np.std(scores)))


    print('scores: ', scores_no_smote, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores_no_smote), np.std(scores_no_smote)))


# read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


# drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

# drop features without information
#features.drop(['feature_2', 'feature_12'], axis=1, inplace=True)

features, labels = removeOutlier(features, labels)
if __name__ == '__main__':
    Run(features, labels)
