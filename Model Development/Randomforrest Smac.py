#this file is only executable in linux
import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTENC,ADASYN,SVMSMOTE,KMeansSMOTE,BorderlineSMOTE, SMOTE
from ConfigSpace import ConfigurationSpace, Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.metrics import accuracy_score
import numpy as np
from Custom_Methods import*
def Run(Xtrain,ytrain):

    #define model parameters
    randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='gini', min_samples_split=3, n_jobs=-1, verbose=False)


    # split the data
    randomforest.fit(Xtrain, ytrain)


    #make Prediction
    #1 dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_downsamp_IMBLearn_NearMiss')
    scores = cross_val_score(randomforest, Xtrain, ytrain, cv=5, scoring='f1_macro', n_jobs=-1)
    #print(scores)
    print('scores: ', scores,"%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)


#drop features without information

#features.drop(['feature_2'], axis=1, inplace=True)

#resample using imbalanced learn
labels=labels.to_numpy().flatten()

#up and down sampling

'''smo = SMOTE(random_state=42)
X_res, y_res = smo.fit_resample(features, labels)
features =X_res
labels = y_res
labels = pd.DataFrame(labels)'''

#convert to numpyarray
#features=features.to_numpy()
#labels=labels.to_numpy().flatten()

def train(config: Configuration, seed: int) -> float:
    model = RandomForestClassifier(n_estimators=config["n_estimators"],max_leaf_nodes=config["max_leaf_nodes"],max_depth=config["max_depth"], random_state=seed,verbose=False,
                                       max_features=config["max_features"], criterion=config["criterion"], min_samples_split= config["min_samples_split"], min_samples_leaf = config["min_samples_leaf"], min_weight_fraction_leaf=config["min_weight_fraction_leaf"])
    model.fit(features, labels)

    scores =stratified_cross_fold_validator_for_smote ( features, labels,5, model)

    return 1- np.mean(scores)

configspace = ConfigurationSpace({"n_estimators" : (10, 300),
                                  "max_depth" : (1,1500),
                                  "random_state" : 211,
                                  "criterion" : ["gini", "entropy", "log_loss"],
                                  "min_samples_split" : (2,1000),
                                  "min_samples_leaf" : (1,200),
                                  "min_weight_fraction_leaf" : (0.0,0.5),
                                  "max_features" : (None, "sqrt", "log2"),
                                  "max_leaf_nodes": (None,400)
                                  })

#scenario defines optimisation env
scenario = Scenario(configspace,deterministic=True, n_trials=1000, n_workers=6,walltime_limit= 3600)

if __name__ == '__main__':
    #use smac for hpo
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
    print(incumbent)
#Run(features, labels)