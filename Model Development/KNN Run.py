from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from sklearn.metrics import f1_score

def csv_to_array_np(csv_file):
    data=np.loadtxt(open(csv_file, "rb"), delimiter=",", skiprows=1)
    arr=np.array(data)
    print(arr.shape)
    print(arr)

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
def runKnn():
    X = np.loadtxt("train_features.csv", delimiter=",", skiprows=1)
    y = np.loadtxt("train_label.csv", delimiter=",", skiprows=1)
    XTest= np.loadtxt("test_features.csv", delimiter=",", skiprows=1)
    yId=y[:,0]
    y=y[:,1]
    model=KNeighborsClassifier(n_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train,y_train)
    pred = model.predict(XTest)
    arrId=[]
    print(len(pred))
    for n in range(len(pred)):
        arrId.append(n)
    print(len(arrId))
    #np.concatenate((yId, pred), axis=1)
    result= [arrId,pred]
    return pred

arr =runKnn()

np.savetxt("result.csv", arr, delimiter=",")
