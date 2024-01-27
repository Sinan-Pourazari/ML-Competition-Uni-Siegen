import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from Custom_Methods import *
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale, RobustScaler
import xgboost as xgb
def Run(Xtrain, ytrain):
    # define model parameters
    '''
    this function contains every parameter, functuion and class to generate a model using Gradient boosting from different librarys
    scaler = RobustScaler()
    Xtrain= scaler.fit_transform(Xtrain)
    Xtrain = pd.DataFrame(Xtrain, columns=names)'''

    #models using the Scikit learn implementaion with basic but different parameters
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

    #models using Light gradient boosting implemetaion
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

    #modles using Extreme gradient boosting implemetaion
    XGBmodel = xgb.XGBClassifier(tree_method='hist', booster='gbtree', learning_rate=0.1, max_depth=14, n_estimators=155)




    #list with all above defined estimator models
    estimators =[#('Gradient', GBmodel),
                ('LGBM', LGBMmodel),
                 ('LGBM2', LGBMmodel2),
                 ('LGBM3', LGBMmodel3),
                 ('LGBM4', LGBMmodel4),
                 ('Gradient2', GBmodel2),
                 ('Gradient3', GBmodel3),
                 ('Gradient4', GBmodel4),
                 ('XGB', XGBmodel)
                 ]
    #stacking clasifier parameters
    clf = StackingClassifier(estimators=estimators,
                             final_estimator= LogisticRegression(),
                             stack_method='predict', verbose= 100, passthrough=False, cv= 15, n_jobs=-1)



    #fit the stack
    clf.fit(Xtrain,ytrain.to_numpy().flatten())

    test = pd.read_csv('test_features.csv')

    #make predictions
    pred = clf.predict(test[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13']])
    # this section was used to evaluated diferen configurations
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, clf, num_workers=10)
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 10, clf, num_workers=10)
    # 6 8330815
    # 8 8299
    print('scores smote: ', scores,
          "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores), np.std(scores)))

    print('scores: ', scores_no_smote,
          "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores_no_smote), np.std(scores_no_smote)))

    return pred


if __name__ == '__main__':
    # read csv
    features = pd.read_csv('train_features.csv')
    labels = pd.read_csv('train_label.csv')

    # drop the id
    features = features.drop(['Id'], axis=1)
    labels = labels.drop(['Id'], axis=1)

    # drop features without information
    # features.drop(['feature_2', 'feature_12'], axis=1, inplace=True)

    features = cap_Outlier(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13']])
    result = Run(features, labels)

    idarr = np.array([])
    for i in range(len(result)):
        idarr = np.append(idarr, i)
    # make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
    return_value = pd.DataFrame({'Id': idarr, 'label': result})
    return_value = return_value.astype(int)
    print(return_value)
    # save it as file
    return_value.to_csv('Enseble_GradienBoosterstemp.csv', columns=['Id', 'label'], index=False)



