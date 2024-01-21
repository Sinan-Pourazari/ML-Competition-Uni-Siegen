from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from sklearn.metrics import f1_score

X = np.loadtxt("train_features.csv", delimiter=",", skiprows=1)
y = np.loadtxt("train_label.csv", delimiter=",", skiprows=1)
y = y[:, 1]
X = [row[1:] for row in X]
print(X)
model = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
f1 = f1_score(y_test, pred, average="macro")

print(f1)