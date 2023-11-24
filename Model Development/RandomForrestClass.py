from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier


class RdmForestClass:

    def __int__(self):
        self.randomforest=RandomForestClassifier(random_state=211, n_estimators=100, criterion='entropy', min_samples_split=2)


    def Run(X,y):
        # split the data
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

        # scale
        #Xtrain = scale(Xtrain)
        #Xtest = scale(Xtest)

        #build randomforrest
        randomforest.fit(Xtrain, ytrain)

        #make Prediction
        pred =randomforest.predict(Xtest)

        score = f1_score(ytest, pred)
        return score








