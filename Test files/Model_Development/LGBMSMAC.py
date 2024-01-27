#this file is only executable in linux
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTENC,ADASYN,SVMSMOTE,KMeansSMOTE,BorderlineSMOTE, SMOTE
from ConfigSpace import ConfigurationSpace, Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.metrics import accuracy_score
import numpy as np
from Custom_Methods import*



#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


features = features[['feature_24', 'feature_16', 'feature_9', 'feature_11', 'feature_8', 'feature_10', 'feature_26', 'feature_2', 'feature_6', 'feature_19']]
labels = labels.drop(['Id'], axis=1)

features, labels = removeOutlier(features,labels)



#resample using imbalanced learn
labels=labels.to_numpy().flatten()

#up and down sampling

smo = SMOTE(random_state=42)
X_res, y_res = smo.fit_resample(features, labels)
features =X_res
labels = y_res
labels = pd.DataFrame(labels)

#convert to numpyarray
#features=features.to_numpy()
#labels=labels.to_numpy().flatten()

def train(config: Configuration, seed: int) -> float:
    model = LGBMClassifier(n_estimators=config["n_estimators"],learning_rate=config["learning_rate"],max_depth=config["max_depth"], random_state=211
                           ,boosting_type=config["boosting_type"], num_leaves=config["num_leaves"], objective='binary', min_split_gain=config["min_split_gian"]
                           , min_child_weight=config["min_child_weight"], min_child_samples=config["min_child_samp"], colsample_bytree=config["colsample_bytree"]
                           , reg_alpha=config["reg_alpha"], reg_lambda=config["reg_lambda"],importance_type=config["importance_type"]
                           )

    model.fit(features, labels)

    scores = cross_val_score(model, features, labels, cv=3, n_jobs=-1)

    return 1- np.mean(scores)

configspace = ConfigurationSpace({
                                  "boosting_type" : ['gbdt', 'dart', 'rf'],
                                  "num_leaves" : (1,500),
                                  "n_estimators" : (10, 10000),
                                  "learning_rate" : (0.01, 0.9),
                                  "min_split_gain" : (0.0),
                                  "min_child_weight" : (0.0000001, 0.001),
                                  "min_child_samp" : (2,500),
                                  "colsample_bytree" : (0.01, 1.0),
                                  "reg_alpha" : (0.0,2.0),
                                  "reg_lambda" : (0.0,2.0),
                                  "importance_type" : ['split', 'gain'],
                                  "max_depth" : (1,3000),

                                  })

#scenario defines optimisation env
scenario = Scenario(configspace,deterministic=True, n_trials=1000, n_workers=6,walltime_limit= 3600)

if __name__ == '__main__':
    #use smac for hpo
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
    print(incumbent)
#Run(features, labels)