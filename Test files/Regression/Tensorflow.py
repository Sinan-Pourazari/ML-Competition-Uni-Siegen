import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


'''config = tf.compat.v1.ConfigProto
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)'''

# Load features and labels from separate CSV files
features_file_path = 'train_features.csv'
labels_file_path = 'train_label.csv'


features = pd.read_csv(features_file_path)
labels = pd.read_csv(labels_file_path)

'''features['timestamp'] = pd.to_datetime(features['timestamp'], unit='ms')

# Convert Unix timestamps to numeric values
features['timestamp_numeric'] = features['timestamp'].astype('int64')

# Normalize the numeric timestamp column
scaler = MinMaxScaler()
features['timestamp_normalized'] = scaler.fit_transform(features[['timestamp_numeric']])

# Drop the intermediate numeric timestamp column if needed
features.drop(columns=['timestamp_numeric','timestamp'],inplace=True, axis=1)

features.rename({"timestamp_normalized": "timestamp"}, inplace=True, axis='columns')
features['timestamp'] = features['timestamp'].astype(float)'''
features['timestamp'] = round(features['timestamp'] / 86400000)
features['timestamp'] = features['timestamp'].astype(int)
print(features)
# Merge features and labels based on 'item', 'user', and 'timestamp'
data = pd.merge(features, labels, on='Id')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the model
embedding_size = 32

user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
item_input = tf.keras.layers.Input(shape=(1,), name='item_input')
timestamp_input = tf.keras.layers.Input(shape=(1,) , name='timestamp_input')

# emedding
user_embedding = tf.keras.layers.Embedding(input_dim=np.max(data['user'] + 1), output_dim=embedding_size)(user_input)
item_embedding = tf.keras.layers.Embedding(input_dim=np.max(data['item'] + 1), output_dim=embedding_size)(item_input)
#timestamp_embedding = tf.keras.layers.Flatten()(timestamp_input)
timestamp_embedding = tf.keras.layers.Embedding(input_dim=np.max(data['timestamp']) + 1, output_dim=embedding_size)(timestamp_input)

concatenated = tf.keras.layers.Concatenate()([user_embedding, item_embedding, timestamp_embedding])

output = tf.keras.layers.Dense(1)(concatenated)

model = tf.keras.models.Model(inputs=[user_input, item_input, timestamp_input], outputs=output)

# Compile the model
model.compile(optimizer='Adagrad', loss='mean_squared_error')

# Train the model
model.fit([train_data['user'], train_data['item'], train_data['timestamp']], train_data['rating'], epochs=1000, batch_size=10240, validation_split=0.2)

# Make predictions
test_predictions = model.predict([test_data['user'], test_data['item'], test_data['timestamp']])
#compute rmse
test_rmse = np.sqrt(mean_squared_error(test_data['rating'], test_predictions.flatten()))

print(f'Test RMSE: {test_rmse}')
#917, 500