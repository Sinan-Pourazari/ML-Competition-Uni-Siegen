import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from Custom_Methods import *
def Run(Xtrain,ytrain):

    #define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=False,
                                         loss='log_loss', criterion='friedman_mse')
    # split the data
    selected = sequential_feature_eliminator(Xtrain, ytrain, GBmodel, verbose=True, remove_outlier=True )
    GBmodel.fit(Xtrain, ytrain.to_numpy().flatten())
    print(selected)
    #make Prediction
    #1 dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_downsamp_IMBLearn_NearMiss')
    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, GBmodel)
    #print(scores)
    print('scores: ', scores,"%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))




if __name__ == '__main__':
    # read csv
    features = pd.read_csv('train_features.csv')
    labels = pd.read_csv('train_label.csv')

    # drop the id
    features = features.drop(['Id', 'feature_2'], axis=1)
    labels = labels.drop(['Id'], axis=1)
    Run(features[['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_0', 'feature_12', 'feature_13', 'feature_9', 'feature_10', 'feature_26', 'feature_4', 'feature_25', 'feature_29', 'feature_27', 'feature_22']],labels)





