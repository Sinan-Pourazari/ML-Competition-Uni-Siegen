import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sn
import datetime as dt

#read CSV Files
data = pd.read_csv('train_features.csv')
dataTarget = pd.read_csv('train_label.csv')

#Join Targets into the dataframe in order to lookduplicates
datajoined=data.set_index('Id').join(dataTarget.set_index('Id'),rsuffix='Feat')
groupedData= datajoined.groupby(data.columns.tolist(),as_index=False).size()
groupedData=groupedData[groupedData['size']>1]
print(len(groupedData))
print(groupedData)

#use Grouping to look how many reviews each user made
dataRev = data.groupby

#Read the features and their names into a normal array for easy acces via indexing
feature_list = []
feature_StringList = []
for feature_name, feature_values in data.items():
    feature_list.append(feature_values)
    feature_StringList.append(feature_name)

#calculate some high level values
for i in range(len(feature_list)):
    print(feature_StringList[i])
    print('median: ', feature_list[i].median())
    print('mean: ', feature_list[i].mean())
    print('standart deviation: ', np.std(feature_list[i]))
    print('min/max: ', min(feature_list[i]), max(feature_list[i]))
    print('')

#plot the plot
fig, big = plt.subplots()
big.scatter(feature_list[1], feature_list[1], label=feature_StringList[1], s=0.1)
big.set(xlabel='user')
plt.legend()
big.grid()

#Histograms
fig, subplot_list = plt.subplots(3,1)
for i in range(len(subplot_list)):
    subplot_list[i].hist(feature_list[i+1],bins=800)
    subplot_list[i].set(title=feature_StringList[i+1])
    subplot_list[i].margins(x=0)

#boxplots
fig, subplot_list=plt.subplots(1, 3)
for i in range(len(subplot_list)):
    subplot_list[i].boxplot(feature_list[i+1])
    subplot_list[i].set(title=feature_StringList[i+1])

plt.show()

# unique Values
#print(data.nunique(axis=0, dropna=False))

#### Correlation matrix ####

corr_matrix = data.corr()
sn.heatmap(corr_matrix, annot=True, cmap='Blues')

plt.show()

