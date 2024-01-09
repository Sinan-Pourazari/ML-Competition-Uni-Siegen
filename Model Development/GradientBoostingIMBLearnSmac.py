#this file is only executable in linux
import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTENC,ADASYN,SVMSMOTE,KMeansSMOTE,BorderlineSMOTE, SMOTE
from ConfigSpace import ConfigurationSpace, Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.metrics import accuracy_score
import numpy as np
def Run(Xtrain,ytrain):

    #define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=True,
                                        loss='log_loss', criterion='friedman_mse')

    # split the data
    GBmodel.fit(Xtrain, ytrain)


    #make Prediction
    #1 dump(GBmodel, 'Model versions for Kaggle Submit/GBmodel6_wo_f2_downsamp_IMBLearn_NearMiss')
    scores = cross_val_score(GBmodel, Xtrain, ytrain, cv=5, scoring='f1_macro', n_jobs=-1)
    #print(scores)
    print('scores: ', scores,"%0.2f F1-Macro with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)


#drop features without information

features.drop(['feature_2'], axis=1, inplace=True)

#resample using imbalanced learn
labels=labels.to_numpy().flatten()

#up and down sampling

smo = SMOTE(random_state=42)
X_res, y_res = smo.fit_resample(features, labels)
features =X_res
labels = y_res
labels = pd.DataFrame(labels)

#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

def train(config: Configuration, seed: int) -> float:
    model = GradientBoostingClassifier(n_estimators=config["n_estimators"],learning_rate=config["learning_rate"],max_depth=config["max_depth"], random_state=config["random_state"],verbose=False,
                                        loss=config["loss"], criterion=config["criterion"], min_samples_split= config["min_samples_split"], min_samples_leaf = config["min_samples_leaf"], min_weight_fraction_leaf=config["min_weight_fraction_leaf"])
    model.fit(features, labels)

    scores = cross_val_score(model, features, labels, cv=3, n_jobs=-1)

    return 1- scores.mean()

configspace = ConfigurationSpace({"n_estimators" : (10, 300),
                                  "learning_rate" : (0.05, 0.4),
                                  "max_depth" : (1,300),
                                  "random_state" : 211,
                                  "loss" : ["log_loss", "exponential"],
                                  "criterion" : ["friedman_mse", "squared_error"],
                                  "min_samples_split" : (2,300),
                                  "min_samples_leaf" : (1,100),
                                  "min_weight_fraction_leaf" : (0.0,0.5)
                                  })

#scenario defines optimisation env
scenario = Scenario(configspace,deterministic=True, n_trials=1000, n_workers=6,walltime_limit= 3600)

if __name__ == '__main__':
    #use smac for hpo
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
    print(incumbent)
#Run(features, labels)