import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train_features.csv')

Id= data['Id']
f0= data['feature_0']
f1 = data['feature_1']
f2 = data['feature_2']
f3 = data['feature_3']
f4 = data['feature_4']
f5 = data['feature_5']
f6 = data['feature_6']
f7 = data['feature_7']
f8 = data['feature_8']

fig_f0,af0 = plt.subplots()
af0.scatter(Id,f0, s=1)
af0.set(xlabel='id', ylabel='f0', title='f0')
af0.grid()

plt.show()
