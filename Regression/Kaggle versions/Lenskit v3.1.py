import lenskit
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn, svd
from lenskit import topn
from lenskit.metrics.predict import rmse
import pandas as pd
import numpy as np

def eval(algo, train, test):
    fittable = util.clone(algo)
    fittable.fit(train)
    users = test.user.array
    items = test.item.array
    #get ratings for NaN replacement
    ratings = train.rating.array
    print(type(ratings))
    average_rating = np.bincount(ratings).argmax()
    print(average_rating)
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
    pred = np.nan_to_num(pred,copy=True, nan=round(average_rating))


    ret = [round(x) for x in pred]
    return ret


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test = pd.read_csv('test_features.csv')


data = pd.merge(features, labels, on='Id')



#drop the id
data = data.drop(['Id'], axis=1)
test.drop(['Id'], axis=1, inplace=True)
#drop duplicate rows
data = data[data['timestamp']>= 1350000000000]

data.drop_duplicates(keep='first', inplace=True)


algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50, iterations=30, reg= 0.2, damping= 4, bias= True)
algo_svd = svd.BiasedSVD(200, damping=2, bias=True)


if __name__ == '__main__':


    #result = eval( algo_ii, data, test)
    result=eval( algo_svd, data, test)

    idarr = np.array([])
    for i in range(len(result)):
        idarr = np.append(idarr, i)
    # make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
    return_value = pd.DataFrame({'Id': idarr, 'Predicted': result})
    return_value = return_value.astype(int)
    print(return_value)
    # save it as file
    return_value.to_csv('Lenskit_BiasedSVD3.csv', columns=['Id', 'Predicted'], index=False)



