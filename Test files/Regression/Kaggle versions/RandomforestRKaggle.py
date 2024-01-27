from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor



def Run(Xtrain,ytrain,Xtest):
    # build randomforrest
    randomforest = RandomForestRegressor(random_state=211, n_estimators=100, min_samples_split=10, verbose=True,
                                         max_features=None, n_jobs=-1, criterion='friedman_mse')

    randomforest.fit(Xtrain, ytrain)

    # make Prediction
    pred = randomforest.predict(Xtest)
    return pred

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test_features = pd.read_csv('test_features.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
test_features = test_features.drop(['Id'],axis=1)




#convert to numpyarray
features=features.to_numpy()
test_features=test_features.to_numpy()
labels=labels.to_numpy().flatten()

result = Run(features,labels,test_features)

#export as csv file
#loop that makes the id for the predicited valus
idarr = np.array([])
for i in range(len(result)):
   idarr= np.append(idarr, i)
#make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
return_value=pd.DataFrame({'Id': idarr, 'Predicted': result})
return_value=return_value.astype(int)
print(return_value)
#save it as file
return_value.to_csv('RDMForR2.csv', columns=['Id', 'Predicted'], index=False)
