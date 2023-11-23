#Linear_SVC. finds a linear seperator which is used to classify the data into two classes
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier

def Run(Xtrain, ytrain, Xtest ):
    # scale
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)

    #make model
    MLPC = MLPClassifier(random_state=211,max_iter=99999, learning_rate_init=0.0041,
                         solver='sgd', learning_rate='adaptive', activation='relu',
                         shuffle= True, n_iter_no_change=3, momentum= 0.9, nesterovs_momentum= True)
    MLPC.fit(Xtrain,ytrain)
    pred = MLPC.predict(Xtest)
    # calculate f1-score
    return pred


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
predFeatures = pd.read_csv('test_features.csv')

#drop the id
features = features.drop(['Id'], axis=1)
labels = labels.drop(['Id'], axis=1)
predFeatures = predFeatures.drop(['Id'], axis=1)

#convert to numpyarray
features=features.to_numpy()
labels=labels.to_numpy().flatten()
predFeatures = predFeatures.to_numpy()

#make the bodel
result = Run(features,labels, predFeatures)

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
return_value.to_csv('MLPC3.csv', columns=['Id', 'label'], index=False)