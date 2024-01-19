import lenskit
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit.algorithms import Recommender, als, item_knn as knn, svd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test = pd.read_csv('test_features.csv')

#megre to remove all duplicates
train = pd.merge(features, labels, on='Id')

#drop duplicate rows
train.drop_duplicates(keep='first', inplace=True)



#drop the id
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

#basic algorithms
algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)
model_before_2010 = als.BiasedMF(features=100, iterations=30, reg=0.15, damping=10, bias=True, method='lu')
model_after_2010 = als.BiasedMF(features=100, iterations=30, reg=0.15, damping=10, bias=True, method='lu')



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


    result = [round(x) for x in pred]

    idarr = np.array([])
    for i in range(len(result)):
        idarr = np.append(idarr, i)
    # make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
    return_value = pd.DataFrame({'Id': idarr, 'Predicted': result})
    return_value = return_value.astype(int)
    print(return_value)
    # save it as file
    return_value.to_csv('Lenskit_BiasedMF4.1.csv', columns=['Id', 'Predicted'], index=False)


