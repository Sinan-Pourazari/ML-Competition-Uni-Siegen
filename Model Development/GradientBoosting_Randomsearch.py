import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import scale
from joblib import dump, load
from sklearn.utils import resample, compute_sample_weight
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif,mutual_info_classif
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
def Run(Xtrain,ytrain,Xtest,ytest):

    #define model parameters
    GBmodel= GradientBoostingClassifier()

    weights = compute_sample_weight(class_weight='balanced', y=ytrain)

    '''    for i in range(len(weights)):
        if (weights[i] > 1):
            weights[i] = weights[i] * 2'''

    param_dist = {
        'loss' : ['log_loss', 'exponential'],
        'criterion' : ['friedman_mse', 'squared_error'],
        'n_estimators': randint(50, 1200),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 2, 0.25, 0.3],
        'max_depth': randint(3, 200),
        'min_samples_split': randint(2, 40),
        'min_samples_leaf': randint(1, 40),
        #'min_weight_fraction_leaf' : [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'random_state' : [42],
        'max_features' : [None, 1, 'sqrt', 'log2'],
        #'n_iter_no_change' : [None, 20, 30, 50, 80, 100, 1000]


    }

    # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        GBmodel,
        param_distributions=param_dist,
        n_iter=4000,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose = 3
    )

    # Fit the RandomizedSearchCV object to the data
    # X_train, y_train are your training data and labels
    #random_search.set_params(weights)
    random_search.fit(X=Xtrain, y= ytrain)

    # Print the best parameters and the corresponding accuracy
    print("Best Parameters: ", random_search.best_params_)
    print("Best Accuracy: ", random_search.best_score_)



#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

features.drop(['Id'], axis=1, inplace=True)
labels.drop(['Id'], axis=1, inplace=True)


#train test split before resampling to get "pure" test data
Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1, random_state=792, stratify=labels)





#convert to numpyarray
Xtrain=Xtrain.to_numpy()
ytrain=ytrain.to_numpy().flatten()
ytest = ytest.to_numpy().flatten()
Xtest = Xtest.to_numpy()



print(Run(Xtrain,ytrain,Xtest,ytest))