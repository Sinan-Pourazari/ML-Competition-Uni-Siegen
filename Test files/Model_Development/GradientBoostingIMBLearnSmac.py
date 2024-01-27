#this file is only executable in linux
import pandas as pd
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


features = features[['feature_24', 'feature_16', 'feature_19', 'feature_11', 'feature_26', 'feature_9', 'feature_0', 'feature_10', 'feature_30', 'feature_25']]
labels = labels[['feature_24', 'feature_16', 'feature_19', 'feature_11', 'feature_26', 'feature_9', 'feature_0', 'feature_10', 'feature_30', 'feature_25']]

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
    model = GradientBoostingClassifier(n_estimators=config["n_estimators"],learning_rate=config["learning_rate"],max_depth=config["max_depth"], random_state=seed,verbose=False,
                                        loss=config["loss"], criterion=config["criterion"], min_samples_split= config["min_samples_split"], min_samples_leaf = config["min_samples_leaf"], min_weight_fraction_leaf=config["min_weight_fraction_leaf"])
    model.fit(features['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13', 'feature_25', 'feature_9', 'feature_6', 'feature_4', 'feature_14', 'feature_28', 'feature_15', 'feature_26'], labels)

    scores = cross_val_score(model, features['feature_24', 'feature_16', 'feature_19', 'feature_17', 'feature_20', 'feature_30', 'feature_29', 'feature_10', 'feature_13', 'feature_25', 'feature_9', 'feature_6', 'feature_4', 'feature_14', 'feature_28', 'feature_15', 'feature_26'], labels, cv=3, n_jobs=-1)

    return 1- np.mean(scores)

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