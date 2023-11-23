#Linear_SVC. finds a linear seperator which is used to classify the data into two classes
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


#basic execution of LSVC algorithm, returns f1-Score on Training data
def LSVC(X,y):
    #split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.33, random_state=42)

    # scale
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)

    #define 'settings' of algorithm
    algo = LinearSVC(dual='auto',random_state=42, tol=1e-5)
    #give data to the algorithm
    algo.fit(Xtrain,ytrain)

    #make predictions
    pred = algo.predict(Xtest)

    #calculate f1-score
    score = f1_score(ytest,pred)
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

print(labels)
print(features)


print(LSVC(features,labels))

