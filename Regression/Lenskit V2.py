import math

from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
from lenskit.metrics.predict import rmse
import pandas as pd
import numpy as np


def Run(train, test):
    fittable = util.clone(als.BiasedMF(50))
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user

    recs = []
    # run recomender
    for i in range(len(users)):
        temp = fittable.recommend(users[i], n=1)
        recs.append(temp['item'])
        fittable.predict()

    ret = pd.DataFrame(recs)
    ret.columns = ['label']

    print(ret)

    return ret['label']


# read csv
features = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_label.csv')
test = pd.read_csv('test_features.csv')

data = pd.merge(features, labels, on='Id')

# drop the id
data = data.drop(['Id'], axis=1)
test.drop(['Id'], axis=1, inplace=True)
print(data)

#data.columns = ['rating', 'user', 'timestamp', 'item']

algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)

result = Run(data, test)
result= result.fillna(5)

print(result)
print(result.value_counts())
idarr = np.array([])
for i in range(len(result)):
    idarr = np.append(idarr, i)
# make pd dataframe with id as axis 0 and the rusulst as label 1 with the results
return_value = pd.DataFrame({'Id': idarr, 'Predicted': result})
return_value = return_value.astype(int)
print(return_value)
# save it as file
return_value.to_csv('Lenskit_BiasedMF2.csv', columns=['Id', 'Predicted'], index=False)
