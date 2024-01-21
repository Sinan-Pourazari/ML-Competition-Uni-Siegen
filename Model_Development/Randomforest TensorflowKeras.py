import tensorflow_decision_forests as tfdf
import pandas as pd

dataset_feature = pd.read_csv('train_features.csv')
dataset_label = pd.read_csv('train_label.csv')

dataset = pd.merge(dataset_label,dataset_label, on='Id')
dataset.drop(['Id'], axis=1, inplace=True)

print(dataset)

tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="label")

model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)

print(model.summary())