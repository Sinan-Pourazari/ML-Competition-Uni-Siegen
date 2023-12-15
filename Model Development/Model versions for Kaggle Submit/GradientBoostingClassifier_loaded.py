from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

def Run(Xtrain,ytrain, Xtest):


    # scale
    #Xtrain = scale(Xtrain)
    #Xtest = scale(Xtest)

    #define model parameters
    GBmodel= load('GBmodel6_wo_f2_downsamp_IMBLearn_editedNearestNeigh')


    # split the data



    #make Prediction
    pred =GBmodel.predict(Xtest)

    return pred


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test_features =pd.read_csv('test_features.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
test_features = test_features.drop(['Id'],axis=1)
#drop features without information
test_features = test_features.drop(['feature_2'],axis=1)
features = features.drop(['feature_2'],axis=1)


#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()




result = Run(features,labels, test_features)

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
return_value.to_csv('GBC9.csv', columns=['Id', 'label'], index=False)