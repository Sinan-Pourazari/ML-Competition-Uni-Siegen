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
    Run(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13', 'feature_25', 'feature_9', 'feature_6', 'feature_4', 'feature_14', 'feature_28', 'feature_15', 'feature_26']], labels)
#0.8335005 F1-Macro with a standard deviation of 0.028
#scores:  [0.86276454 0.80870163 0.8669774  0.8189689  0.838672  ] 0.8392169 F1-Macro with a standard deviation of 0.023
#['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13']
#['feature_0', 'feature_1', 'feature_4', 'feature_5', 'feature_8', 'feature_10', 'feature_11', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_21', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_30']