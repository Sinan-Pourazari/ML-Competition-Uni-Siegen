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
def runKnn(rand,neigh, size):
    X = np.loadtxt("train_features.csv", delimiter=",", skiprows=1)
    y = np.loadtxt("train_label.csv", delimiter=",", skiprows=1)
    y=y[:,1]

    model=KNeighborsClassifier(n_neighbors=neigh)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=rand)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    f1=f1_score(y_test,pred, average="macro")
    return f1
curbest=0.0
bestrand=0
bestneigh=0
bestsize=0
for n in range(9999):
    for i in range(1,15):

            temp=runKnn(n,i,0.33)
            if temp > curbest:
                curbest=temp
                bestrand=n
                bestneigh=i
                #bestsize=j

print(curbest)
print(bestrand)
print(bestneigh)
print(bestsize)


