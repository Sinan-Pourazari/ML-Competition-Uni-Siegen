from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample

def run_resample(feat, target, resample_rating, resample_size):
    '''Hours wasted on trying to fix this when called more than once: 3.5'''
    # match features with labesl for up/down sampling
    datajoined = feat.join(target, on='Id', rsuffix='Feat')
    # extract label to modify
    selected_sample = datajoined.loc[datajoined['rating'] == resample_rating]
    # resample to combat inbalance
    resampled_array = resample(selected_sample, random_state=1, n_samples=resample_size)
    resampled_array.to_csv('debug_resampler pred', index=False)
    # remove all rows with parameter rating
    datajoined.drop(datajoined.loc[datajoined['rating'] == resample_rating].index, inplace=True)
    # add downsampled rows with parameter rating
    datajoined = pd.concat([datajoined, resampled_array])

    # see how many of which lables exist now
    groupedData = datajoined.groupby(['rating'], as_index=False).size()
    print(groupedData)

    # split the features and lables into their own arrays again
    temp_labels = pd.DataFrame(datajoined[['Id', 'rating']])
    temp_features = pd.DataFrame(datajoined[['Id', 'item', 'user', 'timestamp']])
    return temp_features, temp_labels

def run_resample_pandas(feat, target, resample_rating, resample_size):
    # match features with labesl for up/down sampling
    datajoined = feat.join(target, on='Id', rsuffix='Feat')
    print(type(datajoined))
    datajoined.drop(['IdFeat'], axis=1)
    # resample to combat inbalance
    resampled_array = datajoined.resample(label='right')
    resampled_array.to_csv('debug_resampler pred', index=False)

    # see how many of which lables exist now
    groupedData = datajoined.groupby(['rating'], as_index=False).size()
    print(groupedData)

    # split the features and lables into their own arrays again
    temp_labels = pd.DataFrame(datajoined[['Id', 'rating']])
    temp_features = pd.DataFrame(datajoined[['Id', 'item', 'user', 'timestamp']])
    return temp_features, temp_labels

def Run(X,y):
    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

    # scale
   # Xtrain = scale(Xtrain)
    #Xtest = scale(Xtest)


    #make model
    MLP = MLPRegressor(random_state=211,max_iter=20000, solver='adam', learning_rate='adaptive', learning_rate_init=0.004, verbose=True,
                       n_iter_no_change=20,tol=1e-4, activation='logistic', batch_size=700)
    #begin training
    MLP.fit(Xtrain,ytrain)

    #make prediction
    pred = MLP.predict(Xtest)

    # calculate RMSE
    score =np.sqrt(mean_squared_error(ytest, pred))
    temp = pd.DataFrame({'Predicted': pred})

    temp.to_csv('train pred', columns=['Predicted'], index=False)

    return score


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#Resample the data

#features, labels = run_resample_pandas(features, labels, 5, 250000)
#features, labels = run_resample(features, labels, 3, 160000)
#features, labels = run_resample(features, labels, 4, 160000)
#features, labels = run_resample(features, labels, 5, 160000)



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