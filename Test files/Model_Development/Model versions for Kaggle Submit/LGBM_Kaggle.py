import lightgbm as lgbm
import pandas as pd
import numpy as np


def Run(Xtrain, ytrain, Xtest):
    feature_names = list(Xtrain)

    # Xtest = scale(Xtest)
    # Xtrain = scale(Xtrain)

    train_data = lgbm.Dataset(Xtrain, label=ytrain, feature_name=feature_names)
    print(train_data)
    # create validation set

    # create validation set
    # validation_data = train_data.create_valid(features)

    # parameters of the model
    param = {'num_leaves': 32, 'objective': 'binary', 'data_sample_strategy': 'bagging',
             'learning_rate': 0.1}
    param['boosting_type'] = 'gbdt'
    param['metric'] = 'binary_logloss'

    # train model
    model = lgbm.train(param, train_data, 36)

    # save model
    model.save_model('LGBM_Model4.txt')

    # make predictions
    ypred = model.predict(Xtest)

    # rounding the values
    ypred = ypred.round(0)
    # converting from float to integer
    ypred = ypred.astype(int)

    # calculate f1 macro score
    return ypred


# Load the data
X = pd.read_csv('train_features.csv')
y = pd.read_csv('train_label.csv')
Xtest = pd.read_csv('test_features.csv')
# drop the id so it isnt used as a training feature
X.drop(['Id'], axis=1, inplace=True)
y.drop(['Id'], axis=1, inplace=True)
Xtest.drop(['Id'], axis=1, inplace=True)

# drop some more stuff
# X.drop(['feature_2'], axis=1, inplace=True)

result = Run(X, y, Xtest)

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
return_value.to_csv('LGBM4.csv', columns=['Id', 'label'], index=False)