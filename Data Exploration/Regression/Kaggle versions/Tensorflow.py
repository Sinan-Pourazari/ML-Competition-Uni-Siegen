import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load training features, training labels, and test features from separate CSV files
train_features_path = '../train_features.csv'
train_labels_path = 'train_labels.csv'
test_features_path = '../test_features.csv'

train_features = pd.read_csv(train_features_path)
train_labels = pd.read_csv(train_labels_path)
test_features = pd.read_csv(test_features_path)

# Merge training features and labels based on 'item', 'user', and 'timestamp'
train_data = pd.merge(train_features, train_labels, on='Id')

# Preprocess the data
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

train_data['user'] = user_encoder.fit_transform(train_data['user'])
train_data['item'] = item_encoder.fit_transform(train_data['item'])

# Define the model
embedding_size = 10

user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
item_input = tf.keras.layers.Input(shape=(1,), name='item_input')

user_embedding = tf.keras.layers.Embedding(input_dim=len(user_encoder.classes_), output_dim=embedding_size)(user_input)
item_embedding = tf.keras.layers.Embedding(input_dim=len(item_encoder.classes_), output_dim=embedding_size)(item_input)

user_flat = tf.keras.layers.Flatten()(user_embedding)
item_flat = tf.keras.layers.Flatten()(item_embedding)

dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_flat, item_flat])

model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=dot_product)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([train_data['user'], train_data['item']], train_data['rating'], epochs=10, batch_size=64, validation_split=0.2)

# Make predictions for the test set
test_data = test_features.copy()

# Preprocess test data
test_data['user'] = user_encoder.transform(test_data['user'])
test_data['item'] = item_encoder.transform(test_data['item'])

# Make predictions
predictions = model.predict([test_data['user'], test_data['item']])
test_data['predicted_rating'] = predictions.flatten()

# Display or save the predictions
print(test_data[['user', 'item', 'predicted_rating']])
