from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from Custom_Methods import*



def Run(X,y,test):

    #make custom randomforest from whch the model starts
    RDMForest = RandomForestClassifier(random_state=211, n_estimators=20, criterion='log_loss', min_samples_split=10, n_jobs=1,
                                       min_impurity_decrease= 0.0, verbose=True)
    GBmodel= GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=True)

    ABC= AdaBoostClassifier( estimator= RDMForest ,n_estimators=200, random_state=211, learning_rate=1, algorithm='SAMME.R')
    #ABC.fit(Xtrain,ytrain.to_numpy().flatten())
    #cross vaidation
    scores = stratified_cross_fold_validator_for_smote(X, y, 5, ABC)
    print("%0.7f F1-Macro with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
    print(scores)



#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)

#Garbage feature removal
features.drop(inplace=True, labels= ['feature_2'], axis=1)






#basic preprossesing

if __name__ == '__main__':
    print(Run(features, labels, 710))


