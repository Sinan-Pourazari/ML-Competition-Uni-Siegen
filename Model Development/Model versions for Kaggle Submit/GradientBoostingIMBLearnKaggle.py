from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
def Run(Xtrain,ytrain,Xtest):

    #define model parameters
    GBmodel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=10, random_state=211,verbose=False,
                                        loss='log_loss', criterion='friedman_mse')



    # split the data
    GBmodel.fit(Xtrain, ytrain)


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
test.drop(['Id', 'feature_2'], axis=1, inplace=True )

#drop features without information
features.drop(['feature_2'], axis=1, inplace=True)

#resample using imbalanced learn
labels=labels.to_numpy().flatten()

#up and down sampling

smo = SMOTE(random_state=42, n_jobs=-1)
X_res, y_res = smo.fit_resample(features, labels)
features =X_res
labels = y_res

#convert to numpyarray
features=features.to_numpy()
#labels=labels.to_numpy().flatten()


#print(Run(features, labels))
result = Run(features, labels, test)

idarr = np.array([])
for i in range(len(result)):
    idarr = np.append(idarr, i)
# make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
return_value = pd.DataFrame({'Id': idarr, 'Predicted': result})
return_value = return_value.astype(int)
print(return_value)
# save it as file
return_value.to_csv('GBC41.csv', columns=['Id', 'Predicted'], index=False)
