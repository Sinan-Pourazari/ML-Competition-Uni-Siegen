import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sn


data = pd.read_csv('train_features.csv')
dataTarget = pd.read_csv('train_label.csv')

#Join Targets inot the dataframe in order to look for corelations between features and labels
data=data.set_index('Id').join(dataTarget.set_index('Id'),rsuffix='Feat')

print(data)
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

#make multiple figures with 5 plots each to visulise the distribution of the data
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

#figure to further enhance/inspect 2 features (changed the features druing data exploration
fig, big = plt.subplots()
big.plot(feature_list[0], feature_list[29], color='g', label=feature_StringList[29])
big.set(xlabel='id')
big.plot(feature_list[0], feature_list[2], color='r', label=feature_StringList[2])
big.plot(feature_list[0], feature_list[4], color='y', label=feature_StringList[4])

plt.legend()
big.grid()


List_position =0
#Histograms
fig, subplot_list = plt.subplots(4,8, layout="constrained", figsize=(34,20))
for i in range(4):
    for j in range(8):
        subplot_list[i][j].hist(feature_list[List_position],bins=100)
        subplot_list[i][j].set(title=feature_StringList[List_position])
        subplot_list[i][j].margins(x=0)
        List_position +=1
plt.savefig('Histogramm.pdf', dpi=1200)

#Boxplot
List_position = 0
box, subplot_list_box = plt.subplots(4, 8, layout='constrained', figsize=(34,20))
for i in range(4):
        for j in range(8):
            subplot_list_box[i][j].boxplot(feature_list[List_position])
            subplot_list_box[i][j].set(title=feature_StringList[List_position])
            subplot_list_box[i][j].margins(x=0)
            List_position+=1
plt.savefig('Boxplot.pdf', dpi=1200)
#plt.show()

# unique Values
print(data.nunique(axis=0, dropna=False))

#### Correlation matrix ####

corr_matrix = data.corr()
corr_plot, corr = plt.subplots(figsize=(34,20), layout='constrained')
sn.heatmap(corr_matrix, annot=True, cmap='Blues')

plt.savefig('Correlation.pdf', dpi=1200)
plt.show()
#used to inspect features with a correlation value of 1
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