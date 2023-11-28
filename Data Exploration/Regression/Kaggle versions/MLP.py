from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder

def Run(Xtrain, ytrain, Xtest):

    # scale
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)

    #make model
    MLP = MLPRegressor(random_state=211,max_iter=1000, solver='sgd', learning_rate='adaptive', learning_rate_init=0.0041)


    MLP.fit(Xtrain,ytrain)
    pred = MLP.predict(Xtest)
    # calculate f1-score
    return pred

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
testFeatures = pd.read_csv('test_features.csv')


#drop the id
features = features.drop(['Id'], axis=1)
testFeatures = testFeatures.drop(['Id'], axis=1)

labels = labels.drop(['Id'], axis=1)

#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()

#basic preprossesing
enc = OrdinalEncoder()

#for what ever reason this decraees the performance by arround 2.6%
selectorVariance= VarianceThreshold()
features = selectorVariance.fit_transform(features)

result = Run(features,labels,testFeatures)
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
return_value.to_csv('MLPR3.csv', columns=['Id', 'Predicted'], index=False)