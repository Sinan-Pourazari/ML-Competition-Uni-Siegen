from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import pandas as pd

def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

data = pd.merge(features, labels, on='Id')



#drop the id
data = data.drop(['Id'], axis=1)

#drop duplicate rows

data.drop_duplicates(keep='first', inplace=True)



ratings = data
print(ratings.head())

algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)

if __name__ == '__main__':

    all_recs = []
    test_data = []
    for train, test in xf.partition_users(ratings[['user', 'item', 'timestamp','rating']], 5, xf.SampleFrac(0.2)):
        test_data.append(test)
        all_recs.append(eval('ItemItem', algo_ii, train, test))
        all_recs.append(eval('ALS', algo_als, train, test))

    all_recs = pd.concat(all_recs, ignore_index=True)
    all_recs.head()

    test_data = pd.concat(test_data, ignore_index=True)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    results.head()

    print(results.groupby('Algorithm').ndcg.mean())



    results.groupby('Algorithm').ndcg.mean().plot.bar()
