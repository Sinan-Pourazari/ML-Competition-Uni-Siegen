from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor

__all__ =['stratified_cross_fold_validator','stratified_cross_fold_validator_for_smote', 'removeOutlier']
#cross validation
def stratified_cross_fold_validator(X, y, folds, model):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    pred_array = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        current_train_fold_y = y.iloc[train_index]
        current_train_fold_X = X.iloc[train_index]
        current_test_fold_y = y.iloc[test_index]
        current_test_fold_X = X.iloc[test_index]
        model.fit(current_train_fold_X, current_train_fold_y.to_numpy().flatten())
        temp_pred = model.predict(current_test_fold_X)
        pred_array.append(f1_score(current_test_fold_y, temp_pred, average="macro"))


from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

def worker(args):
    X, y, train_index, test_index, model = args
    smo = SMOTE(random_state=42)

    current_train_fold_y = y.iloc[train_index]
    current_train_fold_X = X.iloc[train_index]
    current_train_fold_X, current_train_fold_y = smo.fit_resample(current_train_fold_X, current_train_fold_y)
    current_test_fold_y = y.iloc[test_index]
    current_test_fold_X = X.iloc[test_index]
    model.fit(current_train_fold_X, current_train_fold_y.to_numpy().flatten())
    temp_pred = model.predict(current_test_fold_X)
    return f1_score(current_test_fold_y, temp_pred, average="macro")

def stratified_cross_fold_validator_for_smote(X, y, folds, model, num_workers=10):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    pred_array = []
    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    with Pool(num_workers) as pool:
        pred_array = pool.map(worker, args_list)

    return pred_array


def removeOutlier(X, y):
    # outlier detection
    lof = LocalOutlierFactor(n_neighbors=51)
    outliers = lof.fit_predict(X)

    # convert -1 to 0 for boolean indexing
    outliers[outliers == -1] = 0

    # outlier removal
    repeats = len(X)
    for i in range(repeats):
        if outliers[i] == 0:
            X.drop(index=i, axis=0, inplace=True)
            y.drop(index=i, axis=0, inplace=True)
    # return values
    return X, y