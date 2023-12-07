from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor



def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale
    #Xtrain = scale(Xtrain)
    #Xtest = scale(Xtest)

    #build randomforrest
    randomforest=RandomForestRegressor(random_state=211, n_estimators=100, min_samples_split=10,verbose=True, max_features=None, n_jobs=-1,criterion='friedman_mse')
    randomforest.fit(Xtrain, ytrain)

    #make Prediction
    pred =randomforest.predict(Xtest)

    #get RMSE
    score = np.sqrt(mean_squared_error(ytest, pred))

    #output
    temp = pd.DataFrame({'Predicted': pred})

    temp.to_csv('train pred', columns=['Predicted'], index=False)
    return score

#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')


#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)




#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()




print(Run(features,labels))