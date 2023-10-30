from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import csv
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

def outlierDetec(arr):
    q1= np.quantile(arr,0.25)
    q3= np.quantile(arr, 0.75)
    IQR= q3-q1
    upper_lim=q3+(1.5*IQR)
    lower_lim=q1-(1.5*IQR)
    return [lower_lim, upper_lim]

def removeOutlier(arrOrig, target):
    arr=arrOrig
    y=target
    for i in range(len(arr)):
        lim=outlierDetec(arr[i])
        for j in range(len(arr[i])):
            if arr[i,j] < lim[0] or arr[i,j] > lim[1]:
                for r in range(len(arr)):
                    np.delete(arr[r],j)
                    np.delete(y,j)

    return (arr, y)



def csv_to_array_np(csv_file):
    data=np.loadtxt(open(csv_file, "rb"), delimiter=",", skiprows=1)
    arr=np.array(data)
    return arr

def csv_to_array(csv_file):
    with open(csv_file) as file:
        reader = csv.reader(file)

        result= [[] for _ in next(reader)]
        for row in reader:
            for col, value in enumerate(row):
                result[col].append(value)
    return result

#print(csv_to_array("train_features.csv"))

#X=np.array(csv_to_array("train_features.csv"))
#y=np.array(csv_to_array("train_label.csv"))

def normalizeScale(arr):
    temp =arr
    return scale(temp)


def runKnn(X,y):

    y=y[:,1]

    model=KNeighborsClassifier(n_neighbors=23)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print(pred)
    f1=f1_score(y_test,pred, average="macro")
    return f1

#X = np.loadtxt("train_features.csv", delimiter=",", skiprows=1)
#y = np.loadtxt("train_label.csv", delimiter=",", skiprows=1)
X=csv_to_array_np('train_features.csv')
y=csv_to_array_np('train_label.csv')
X,y=removeOutlier(X,y)
X=normalizeScale(X)

print(runKnn(X,y))

