import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train_features.csv')

Id = data['Id']
f0 = data['feature_0']
f1 = data['feature_1']
f2 = data['feature_2']
f3 = data['feature_3']
f4 = data['feature_4']
f5 = data['feature_5']
f6 = data['feature_6']
f7 = data['feature_7']
f8 = data['feature_8']

feature_list = [f0, f1, f2, f3, f4, f5, f6, f7, f8]
feature_StringList = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']

fig, subplot_list = plt.subplots(2, 5, squeeze=True)


for j in range(5):

    temp = subplot_list[0,j]
    temp.scatter(Id, feature_list[j], s=1)
    temp.set(xlabel='id', ylabel=feature_StringList[j], title=feature_StringList[j])
    temp.grid()

for j in range(4):

    temp = subplot_list[1,j]
    temp.scatter(Id, feature_list[j+5], s=0.5)
    temp.set(xlabel='id', ylabel=feature_StringList[j+5], title=feature_StringList[j+5])
    temp.grid()

fig, big = plt.subplots()
big.plot(Id, f1,color='g', label='f1')
big.set(xlabel='id')
big.plot(Id, f3,color='r', label='f3')

plt.show()
