import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from Custom_Methods import *

def Run(Xtrain, ytrain):
    # define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False,
                                        loss='log_loss', criterion='friedman_mse')

    #order= permutation_tester(Xtrain,ytrain,GBmodel,verbose=True)
    #to preven dataleak each train fold is oversampled indepenantly
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, GBmodel)
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 10, GBmodel)

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
    result = (features[ ['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13', 'feature_25']], labels)
