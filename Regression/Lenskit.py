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

    #run recomender
    recs = batch.recommend(fittable, users, 100)
    recs['Algorithm'] = aname

    return recs


#read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')

data = pd.merge(features, labels, on='Id')

#drop the id
data = data.drop(['Id'], axis=1)

algo_ii = knn.ItemItem(20)

all_recs= []
test_data =[]
for train, test in xf.partition_users(data[['user', 'item', 'timestamp']], 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    all_recs.append(eval('ItemItem', algo_ii, train, test))
print(all_recs)