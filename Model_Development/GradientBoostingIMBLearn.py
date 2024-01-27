import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from Custom_Methods import *

def Run(Xtrain, ytrain):
    #Xtrain = cap_Outlier(Xtrain)
    # define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False,
                                        loss='log_loss', criterion='friedman_mse')

    #order= permutation_tester(Xtrain,ytrain,GBmodel,verbose=True)
    #to preven dataleak each train fold is oversampled indepenantly
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 10, GBmodel)

    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, GBmodel)

    print('scores smote: ', scores, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores), np.std(scores)))

    print('scores: ', scores_no_smote, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores_no_smote), np.std(scores_no_smote)))





if __name__ == '__main__':
    # read csv
    features = pd.read_csv('train_features.csv')
    labels = pd.read_csv('train_label.csv')

    # drop the id
    labels = labels.drop(['Id'], axis=1)

    # drop features without information
    # features.drop(['feature_2', 'feature_12'], axis=1, inplace=True)
    features ,labels= removeOutlier(features, labels)

    result = Run(features[['feature_24', 'feature_16', 'feature_27', 'feature_9', 'feature_29', 'feature_20', 'feature_19', 'feature_17', 'feature_25', 'feature_12', 'feature_10', 'feature_4', 'feature_13', 'feature_15']], labels)
