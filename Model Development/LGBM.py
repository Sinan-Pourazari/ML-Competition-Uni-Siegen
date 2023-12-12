import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from lightgbm import LGBMClassifier

def Run(features, labels):
    feature_names = list(features)
    print(feature_names)
    Xtrain, Xtest, ytrain, ytest = train_test_split(features,labels, train_size=0.2, random_state=42)
    Xtrain, Xval, ytrain, yval = train_test_split(X, y, random_state=17, test_size=0.1)

    print(Xtrain)
    #Xtest = scale(Xtest)
    #Xtrain = scale(Xtrain)

    ytest = ytest.to_numpy().flatten()

    train_data = lgbm.Dataset(Xtrain, label=ytrain,    feature_name=feature_names, categorical_feature=['feature_10', 'feature_12'])
    print(train_data)
    #create validation set
    validation_data = lgbm.Dataset(data=Xval, label= yval, reference=train_data, feature_name=feature_names, categorical_feature=['feature_10', 'feature_12'])

    #create validation set
    #validation_data = train_data.create_valid(features)

    #parameters of the model
    param = {'num_leaves': 64, 'objective': 'binary', 'data_sample_strategy': 'bagging',
             'learning_rate': 0.1, 'early_stoping_round': 1}
    param['boosting_type'] = 'gbdt'
    param['metric'] = 'binary_logloss'


    #train model
    model = lgbm.train(param, train_data, 100, valid_sets=[validation_data])

    #save model
    model.save_model('LGBM_Model3.txt')

    #make predictions
    ypred = model.predict(Xtest)

    # rounding the values
    ypred = ypred.round(0)
    # converting from float to integer
    ypred = ypred.astype(int)

    #calculate f1 macro score
    score = f1_score(ytest,ypred, average='macro')
    return score



#Load the data
X = pd.read_csv('train_features.csv')
y = pd.read_csv('train_label.csv')




#drop the id so it isnt used as a training feature
X.drop(['Id'], axis=1, inplace=True)
y.drop(['Id'], axis=1, inplace=True)


#drop some more stuff
#X.drop(['feature_2'], axis=1, inplace=True)
print(Run(X,y))

