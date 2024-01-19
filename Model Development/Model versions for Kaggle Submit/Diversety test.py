import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from Custom_Methods import *
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def Run(Xtrain, ytrain):
    # define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False,
                                        loss='log_loss', criterion='friedman_mse')
    randomforest=RandomForestClassifier(random_state=21, n_estimators=64, criterion='entropy', min_samples_split=2, n_jobs=1,max_depth=1409,
                                        max_features='sqrt', max_leaf_nodes=138, min_samples_leaf=3,verbose=False, min_weight_fraction_leaf=0.00215566094893653)
    #make custom randomforest from whch the model starts
    RDMForest = RandomForestClassifier(random_state=211, n_estimators=20, criterion='log_loss', min_samples_split=10, n_jobs=1,
                                       min_impurity_decrease= 0.0, verbose=False)
    GBmodelada= GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False)

    ABCRDMFR= AdaBoostClassifier( estimator= RDMForest ,n_estimators=200, random_state=211, learning_rate=1, algorithm='SAMME.R')
    ABCGB= AdaBoostClassifier( estimator= GBmodelada ,n_estimators=200, random_state=211, learning_rate=1, algorithm='SAMME.R')



    #order= permutation_tester(Xtrain,ytrain,GBmodel,verbose=True)
    #to preven dataleak each train fold is oversampled indepenantly

    estimators =[('Gradient', GBmodel),
                ('rdm', randomforest),
                 ('AdaGradient', ABCGB),
                 ('AdaRDM', ABCRDMFR)
                 ]
    clf = StackingClassifier(estimators=estimators, final_estimator= GradientBoostingClassifier(), stack_method='predict', verbose= 100, passthrough=False)
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, clf, num_workers=5)
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 10, clf, num_workers=5)

    print('scores smote: ', scores, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores), np.std(scores)))

    print('scores: ', scores_no_smote, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores_no_smote), np.std(scores_no_smote)))



    clf.fit(Xtrain,ytrain.to_numpy().flatten())
    test = pd.read_csv('test_features.csv')
    pred = clf.predict(test[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13']])
    return pred

# read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


# drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

# drop features without information
#features.drop(['feature_2', 'feature_12'], axis=1, inplace=True)

features, labels = removeOutlier(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13']], labels)
if __name__ == '__main__':
    result = Run(features, labels)

    idarr = np.array([])
    for i in range(len(result)):
        idarr = np.append(idarr, i)
    # make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
    return_value = pd.DataFrame({'Id': idarr, 'label': result})
    return_value = return_value.astype(int)
    print(return_value)
    # save it as file
    return_value.to_csv('Enseble_temp.csv', columns=['Id', 'label'], index=False)



