import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from Custom_Methods import *
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale, RobustScaler
import xgboost as xgb
def Run(Xtrain, ytrain):

    #define model parameters for Scikit learn Gradient booster
    GBmodel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False,
                                        loss='log_loss', criterion='friedman_mse')
    GBmodel2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=False,
                                         loss='log_loss', criterion='squared_error')
    GBmodel3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=False,
                                         loss='exponential', criterion='friedman_mse')
    GBmodel4 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=False,
                                         loss='exponential', criterion='squared_error')


    #define parameters for Extreme gradient boosting Clasifier
    XGBmodel = xgb.XGBClassifier(tree_method='hist', booster='gbtree', learning_rate=0.1, max_depth=14, n_estimators=155)
    XGBmodel2 = xgb.XGBClassifier(tree_method='exact', booster='gbtree', learning_rate=0.1, max_depth=14, n_estimators=155)
    XGBmodel3 = xgb.XGBClassifier(tree_method='approx', booster='dart', learning_rate=0.1, max_depth=14, n_estimators=155)
    XGBmodel4 = xgb.XGBClassifier(tree_method='hist', booster='dart', learning_rate=0.1, max_depth=14, n_estimators=155)

    #define model parameters for Light Gradient Boosting classifier
    LGBMmodel = LGBMClassifier(random_state=211, boosting_type='dart', objective='binary', num_leaves=51,
                           learning_rate=0.1, n_estimators=100,
                           max_depth=-1, importance_type='split')
    LGBMmodel2 = LGBMClassifier(random_state=211, boosting_type='gbdt', objective='binary', num_leaves=51,
                               learning_rate=0.1, n_estimators=150,
                               max_depth=-1, importance_type='split')
    LGBMmodel3 = LGBMClassifier(random_state=211, boosting_type='gbdt', objective='binary', num_leaves=51,
                               learning_rate=0.1, n_estimators=100,
                               max_depth=-1, importance_type='gain')
    LGBMmodel4 = LGBMClassifier(random_state=211, boosting_type='dart', objective='binary', num_leaves=51,
                               learning_rate=0.1, n_estimators=100,
                               max_depth=-1, importance_type='gain')

    #list with all estimators
    estimators =[('Gradient', GBmodel),
                ('LGBM', LGBMmodel),
                 ('LGBM2', LGBMmodel2),
                 ('LGBM3', LGBMmodel3),
                 ('LGBM4', LGBMmodel4),
                 ('Gradient2', GBmodel2),
                 ('Gradient3', GBmodel3),
                 ('Gradient4', GBmodel4),
                 ('XGB', XGBmodel)
                ('XGB2', XGBmodel2),
               ('XGB3', XGBmodel3),
                ('XGB4', XGBmodel4)
                 ]
    #Stacking classifier with all above defined estimators using Logistic Regression to stack them
    clf = StackingClassifier(estimators=estimators,
                             final_estimator= LogisticRegression(),
                             stack_method='predict', verbose= 0, passthrough=False, cv= 10)


    #crossfold Vallidators for smote and "natural" dataset
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, clf, num_workers=10)
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 10, clf, num_workers=10)

    #fit the Stacking Classifier to the training data
    clf.fit(Xtrain,ytrain.to_numpy().flatten())

    #load the test data to predict on
    test = pd.read_csv('test_features.csv')

    #generate predictiosn on the same features as the fit data
    pred = clf.predict(test[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20',
                                               'feature_0', 'feature_12', 'feature_13']])
    #this wasused to print the scores of the validators
    print('scores smote: ', scores,
          "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores), np.std(scores)))

    print('scores: ', scores_no_smote,
          "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores_no_smote), np.std(scores_no_smote)))

    #return list with perdictions
    return pred


if __name__ == '__main__':
    # read csv
    features = pd.read_csv('train_features.csv')
    labels = pd.read_csv('train_label.csv')

    # drop the id
    features = features.drop(['Id', 'feature_2'], axis=1)
    labels = labels.drop(['Id'], axis=1)

    # drop features without information
    # features.drop(['feature_2', 'feature_12'], axis=1, inplace=True)

    features = cap_Outlier(features)

    result = Run(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20',
                                               'feature_0', 'feature_12', 'feature_13']], labels)

    idarr = np.array([])
    for i in range(len(result)):
        idarr = np.append(idarr, i)
    # make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
    return_value = pd.DataFrame({'Id': idarr, 'label': result})
    return_value = return_value.astype(int)
    print(return_value)
    # save it as file
    return_value.to_csv('Enseble_GradienBoosterstemp.csv', columns=['Id', 'label'], index=False)



