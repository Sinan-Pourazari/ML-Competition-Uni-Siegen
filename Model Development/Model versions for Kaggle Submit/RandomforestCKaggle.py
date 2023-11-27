from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier



def Run(Xtrain,ytrain,Xtest):

    #build randomforrest
    randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='entropy', min_samples_split=2)
    randomforest.fit(Xtrain, ytrain)

    #make Prediction
    pred =randomforest.predict(Xtest)

    return pred

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
testfeatures = pd.read_csv('test_features.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
testfeatures= testfeatures.drop(['Id'], axis=1)

#remove potential garbage features
features.drop(['feature_0', 'feature_21','feature_12'], axis=1, inplace=True)
testfeatures.drop(['feature_0', 'feature_21', 'feature_12'], axis=1, inplace=True)





#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()
testfeatures.to_numpy()

#basic preprossesing

selectorVariance= VarianceThreshold()
features = selectorVariance.fit_transform(features)
testfeatures=testfeatures.drop(['feature_2'],axis=1)
result =Run(features, labels, testfeatures)

#export as csv file
#loop that makes the id for the predicited valus
idarr = np.array([])
for i in range(len(result)):
   idarr= np.append(idarr, i)
#make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
return_value=pd.DataFrame({'Id': idarr, 'label': result})
return_value=return_value.astype(int)
print(return_value)
#save it as file
return_value.to_csv('RDMForC3.csv', columns=['Id', 'label'], index=False)