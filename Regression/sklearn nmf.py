from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import non_negative_factorization
def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)



    #make model
    nnf = non_negative_factorization(n_components=2, init='random', random_state=0, X)




    nnf.fit(Xtrain,ytrain)
    pred = nnf.predict(Xtest)
    # calculate RMSE
    score =np.sqrt(mean_squared_error(ytest, pred))
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

#basic preprossesing
#enc = OrdinalEncoder()

#for what ever reason this decraees the performance by arround 2.6%
#selectorVariance= VarianceThreshold()
#features = selectorVariance.fit_transform(features)

print("RMSE: ", Run(features, labels))