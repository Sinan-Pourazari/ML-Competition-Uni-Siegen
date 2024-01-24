import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from lightgbm import LGBMClassifier
from Custom_Methods import*

def Run(Xtrain, ytrain):
    #feature_names = list(features)
    #Xtrain, ytrain = removeOutlier(features,labels)



    #train model
    model = LGBMClassifier(n_jobs=-1, random_state=211, boosting_type='dart', objective='binary',num_leaves=51, learning_rate=0.1, n_estimators=100,
                           max_depth=-1, importance_type='split',)

    selected = sequential_feature_selector(Xtrain, ytrain, model, verbose=True, remove_outlier=True )

    #model.fit(Xtrain,ytrain.to_numpy().flatten())

    scores = stratified_cross_fold_validator_for_smote(Xtrain, ytrain, 10, model, num_workers=5)
    scores_no_smote = stratified_cross_fold_validator(Xtrain, ytrain, 10, model, num_workers=5)

    print('scores smote: ', scores, "%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores), np.std(scores)))
    print('scores no smote: ', scores_no_smote ,"%0.7f F1-Macro with a standard deviation of %0.3f" % (np.mean(scores_no_smote), np.std(scores_no_smote)))


#Load the data
X = pd.read_csv('train_features.csv')
y = pd.read_csv('train_label.csv')




#drop the id so it isnt used as a training feature
X.drop(['Id'], axis=1, inplace=True)
y.drop(['Id'], axis=1, inplace=True)


#drop some more stuff
#X.drop(['feature_2'], axis=1, inplace=True)
if __name__ == '__main__':
    print(Run(X,y))

