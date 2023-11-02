import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sn


data = pd.read_csv('train_features.csv')
# datatest = pd.read_csv('test_features.csv')


feature_list = []
feature_StringList = []

for feature_name, feature_values in data.items():
    feature_list.append(feature_values)
    feature_StringList.append(feature_name)

fig, subplot_list = plt.subplots(2, 5, squeeze=True)

# fig1 id to feature 3
for j in range(5):
    temp = subplot_list[0, j]
    temp.scatter(feature_list[0], feature_list[j], s=1)
    temp.set(xlabel='id', ylabel=feature_StringList[j], title=feature_StringList[j])
    temp.grid()
# fig 1 feature 4 to 8
for j in range(5):
    temp = subplot_list[1, j]
    temp.scatter(feature_list[0], feature_list[j + 5], s=0.5)
    temp.set(xlabel='id', ylabel=feature_StringList[j + 5], title=feature_StringList[j + 5])
    temp.grid()

fig, subplot_list = plt.subplots(2, 5, squeeze=True)

# fig 2 feature 9 to 13
for j in range(5):
    temp = subplot_list[0, j]
    temp.scatter(feature_list[0], feature_list[j + 10], s=1)
    temp.set(xlabel='id', ylabel=feature_StringList[j + 10], title=feature_StringList[j + 10])
    temp.grid()
# fig 2 feature 14 to 18
for j in range(5):
    temp = subplot_list[1, j]
    temp.scatter(feature_list[0], feature_list[j + 15], s=0.5)
    temp.set(xlabel='id', ylabel=feature_StringList[j + 15], title=feature_StringList[j + 15])
    temp.grid()

fig, subplot_list = plt.subplots(2, 5, squeeze=True)
# fig 3 feature 19 to 23
for i in range(5):
    temp = subplot_list[0, i]
    temp.scatter(feature_list[0], feature_list[i + 20], s=1)
    temp.set(xlabel='id', ylabel=feature_StringList[i + 20], title=feature_StringList[i + 20])
    temp.grid()
# fig 3 feature 24 to 28
for j in range(5):
    temp = subplot_list[1, j]
    temp.scatter(feature_list[0], feature_list[j + 25], s=0.5)
    temp.set(xlabel='id', ylabel=feature_StringList[j + 25], title=feature_StringList[j + 25])
    temp.grid()

fig, subplot_list = plt.subplots(ncols=2, squeeze=True)
# fig 4 feature 29 to 30
for i in range(2):
    temp = subplot_list[i]
    temp.scatter(feature_list[0], feature_list[i + 30], s=1)
    temp.set(xlabel='id', ylabel=feature_StringList[i + 30], title=feature_StringList[i + 30])
    temp.grid()

fig, big = plt.subplots()
big.plot(feature_list[0], feature_list[10], color='g', label=feature_StringList[10])
big.set(xlabel='id')
big.plot(feature_list[0], feature_list[20], color='r', label=feature_StringList[20])
plt.legend()
big.grid()



plt.show()

# unique Values
print(data.nunique(axis=0, dropna=False))

#### Correlation matrix ####

corr_matrix = data.corr()
sn.heatmap(corr_matrix, annot=True, cmap='Blues')

plt.show()

diff_array=[]
diffdiff_array=[]
arr10=feature_list[10].to_numpy(copy=False)
arr20=feature_list[20].to_numpy(copy=False)
for i in range(len(data)):
    if arr10[i] != arr20[i]:
        diff_array.append([i,arr20[i],arr10[i]])
        diffdiff_array.append(arr20[i]-arr10[i])
print(diff_array)
print(diffdiff_array)
print(max(diffdiff_array),min(diffdiff_array))
print(len(diff_array))