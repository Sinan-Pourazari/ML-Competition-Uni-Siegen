from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_sample_weight


def Run(Xtrain,ytrain,Xtest,weights):


    #define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=211,
                                         verbose=True,
                                         loss='log_loss', criterion='friedman_mse')

    used_weight = []
    for i in ytrain:
        if ytrain[i] is 0:
            used_weight.append(weights[1])
        else:
            used_weight.append((weights[0]))
    # split the data
    GBmodel.fit(Xtrain, ytrain, used_weight)


    #make Prediction
    pred =GBmodel.predict(Xtest)

    return pred

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test = pd.read_csv('test_features.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
test.drop(['Id', 'feature_2', 'feature_1', 'feature_13', 'feature_21'
], axis=1, inplace=True )

#drop features without information
features.drop(['feature_2', 'feature_1', 'feature_13', 'feature_21'
], axis=1, inplace=True)
#resample using imbalanced learn
labels=labels.to_numpy().flatten()

#up and down sampling
#up and down sampling
weights = compute_sample_weight(class_weight='balanced', y=labels)


smo = SMOTE(random_state=42, n_jobs=-1)
X_res, y_res = smo.fit_resample(features, labels)
features =X_res
labels = y_res

#convert to numpyarray
features=features.to_numpy()
#labels=labels.to_numpy().flatten()


#print(Run(features, labels))
result = Run(features, labels, test, weights)

idarr = np.array([])
for i in range(len(result)):
    idarr = np.append(idarr, i)
# make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
return_value = pd.DataFrame({'Id': idarr, 'Predicted': result})
return_value = return_value.astype(int)
print(return_value)
# save it as file
return_value.to_csv('GBC50.csv', columns=['Id', 'Predicted'], index=False)
