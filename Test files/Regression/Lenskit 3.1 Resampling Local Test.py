from collections import Counter

import lenskit
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn, svd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import RandomOverSampler

def eval(algo, train, test):
    fittable = util.clone(algo)
    fittable.fit(train)
    users = test.user.array
    items = test.item.array
    # now we run the recommender
    #recs = batch.recommend(fittable, users, 100)
    recs=[]
    for i in range(len(users)):
        temp = fittable.predict_for_user(users[i], [items[i]])
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
    return ret


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

#megre to remove all duplicates
data = pd.merge(features, labels, on='Id')

#drop duplicate rows
data.drop_duplicates(keep='first', inplace=True)


train, test = train_test_split(data, test_size= 0.1, random_state= 532)

#split it for resampling
labels = train['rating']
features = train.drop(['rating'], axis=1)

#resample to combat imbalance
dist = {
    4: 400000,
    3: 300000,
    2: 300000,
    1: 350000,
}
ros = RandomOverSampler(random_state=42, sampling_strategy=dist)

xtrain, ytrain = ros.fit_resample(features, labels)
print(Counter(ytrain))
xtrain.join(ytrain)
train=xtrain

#drop the id
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

#seperate label from features in test set
test_label = test['rating']
test_feature = test.drop(['rating'], axis=1, inplace = False)

#basic algorithms
algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)
algo_svd = svd.BiasedSVD(50, damping=5, bias=True)


if __name__ == '__main__':
    #result = eval( algo_ii, data, test)
    result=eval( algo_svd, data, test_feature)
    print(np.sqrt(mean_squared_error(test_label, result)))





