import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from lightgbm import LGBMClassifier

def fit(Xtrain, ytrain):
    feature_names = list(Xtrain)

    #Xtest = scale(Xtest)
    #Xtrain = scale(Xtrain)

    ytran = ytrain.to_numpy().flatten()

    train_data = lgbm.Dataset(Xtrain, label=ytrain,    feature_name=feature_names)
    #create validation set
    #validation_data = train_data.create_valid(features)

    #parameters of the model
    param = {'num_leaves': 32, 'objective': 'binary', 'data_sample_strategy': 'bagging',
             'learning_rate': 0.01}
    param['boosting_type'] = 'gbdt'
    param['metric'] = 'binary_logloss'


    #train model
    model = lgbm.train(param, train_data, 100)

    return model

'''def predict(X):
    ypred = model.predict(X)

    # rounding the values
    ypred = ypred.round(0)
    # converting from float to integer
    ypred = ypred.astype(int)'''


#Load the data
X = pd.read_csv('train_features.csv')
y = pd.read_csv('train_label.csv')




#drop the id so it isnt used as a training feature
X.drop(['Id'], axis=1, inplace=True)
y.drop(['Id'], axis=1, inplace=True)


#drop some more stuff
#X.drop(['feature_2'], axis=1, inplace=True)

