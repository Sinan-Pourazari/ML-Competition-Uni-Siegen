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
    """

    :param algo: Defines the lenskit algortihm to be used for the model
    :param train: test set
    :param test: treain set
    :return: predicitons
    """
    #the fittable is the algorithm used it is cloned dude to the recomendation from the getting started. although in this versoin of code it is useless
    fittable = util.clone(algo)

    #the algortim is trained on the train data
    fittable.fit(train)

    #the features from the test set used in the MF process
    users = test.user.array
    items = test.item.array

    #get ratings for NaN replacement
    ratings = train.rating.array

    #replace NAN ratings wtih the rating tahat is most common
    average_rating = np.bincount(ratings).argmax()
    print(average_rating)
    # now run recomender

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

#merge train label an feature into one dataset
data = pd.merge(features, labels, on='Id')

#drop the id
data = data.drop(['Id'], axis=1)
test.drop(['Id'], axis=1, inplace=True)

#filter out too old data
data = data[data['timestamp']>= 1448838000000]

#drop duplicate rows
data.drop_duplicates(keep='first', inplace=True)

#diferent algorithms
algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(features=100, iterations=160, reg=0.15, damping=5, bias=True, method='cd')
algo_svd = svd.BiasedSVD(100, damping=2, bias=True)


if __name__ == '__main__':


    #run the perdiciton
    result=eval( algo_als, data, test)

    #make a retunr csv. this is possibly the most elaboarte way to do this.
    idarr = np.array([])
    for i in range(len(result)):
        idarr = np.append(idarr, i)
    # make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
    return_value = pd.DataFrame({'Id': idarr, 'Predicted': result})
    return_value = return_value.astype(int)
    print(return_value)
    # save it as file
    return_value.to_csv('Lenskit_BiasedMF3.csv', columns=['Id', 'Predicted'], index=False)



