import lenskit
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn, svd
from lenskit import topn
from lenskit.metrics.predict import rmse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import Counter


def train(algo, train):
    fittable = util.clone(algo)
    fittable.fit(train)

    return fittable



#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#megre to remove all duplicates
data = pd.merge(features, labels, on='Id')

#drop duplicate rows
data.drop_duplicates(keep='first', inplace=True)
#split it for the train test split
labels = data['rating']
features = data.drop(['rating'], axis=1)

train, test = train_test_split(data, test_size= 0.1, random_state= 532)


#drop the id
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

#seperate label from features in test set
test_label = test['rating']
test_feature = test.drop(['rating'], axis=1, inplace = False)

#basic algorithms
algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)
model_before_2010 = svd.BiasedSVD(50, damping=5, bias=True)
model_after_2010 = svd.BiasedSVD(50, damping=5, bias=True)



if __name__ == '__main__':
    #result = eval( algo_ii, data, test)
    #split into 2 arrays based on time
    before_2010_train = train[train['timestamp'] <= 1293836400000]
    after_2010_train = train[train['timestamp'] > 1293836400000]
    #make a model for data before 2016 and after 2016
    model_before_2010.fit(before_2010_train)
    model_after_2010.fit(after_2010_train)


    #make predictions
    users = test.user.array
    items = test.item.array
    timestamps = test.timestamp.array
    #recs = batch.recommend(fittable, users, 100)
    recs=[]
    for i in range(len(users)):
        if timestamps[i] <= 1293836400000:
            temp = model_before_2010.predict_for_user(users[i], [items[i]])
            recs.append(temp)
        else:
            temp = model_after_2010.predict_for_user(users[i], [items[i]])
            recs.append(temp)

    #ret = pd.DataFrame(recs)
    #print(ret)
    pred = []
    i=0
    for j in items:
        pred.append(recs[i][j])
        i=i+1
    #remove NaN and repalce with 5 due to the bias
    pred = np.nan_to_num(pred,copy=True, nan=5)


    ret = [round(x) for x in pred]


    print(np.sqrt(mean_squared_error(test_label, ret)))





