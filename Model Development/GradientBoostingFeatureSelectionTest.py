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
    GBmodel.fit(Xtrain, ytrain.to_numpy().flatten())
    best_score, selected = selsect_features_all(Xtrain, ytrain, GBmodel)
    print(best_score, selected)
    #make Prediction
    #1 dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_downsamp_IMBLearn_NearMiss')
    scores = stratified_cross_fold_validator(Xtrain, ytrain, GBmodel)
    #print(scores)
    print('scores: ', scores,"%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
if __name__ == '__main__':
    Run(features,labels)





