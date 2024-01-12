import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.feature_selection import SequentialFeatureSelector

__all__ =['stratified_cross_fold_validator','stratified_cross_fold_validator_for_smote', 'removeOutlier','select_features', 'selsect_features_all']
#needs pandas dataframe to work


def worker_oversample(args):
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

def worker_standart(args):
    X, y, train_index, test_index, model = args

    current_train_fold_y = y.iloc[train_index]
    current_train_fold_X = X.iloc[train_index]
    current_test_fold_y = y.iloc[test_index]
    current_test_fold_X = X.iloc[test_index]
    model.fit(current_train_fold_X, current_train_fold_y.to_numpy().flatten())
    temp_pred = model.predict(current_test_fold_X)
    return f1_score(current_test_fold_y, temp_pred, average="macro")

#needs pandas dataframe to work
def stratified_cross_fold_validator_for_smote(X, y, folds, model, num_workers=10):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    with Pool(num_workers) as pool:
        pred_array = pool.map(worker_oversample, args_list)

    return pred_array

def stratified_cross_fold_validator(X, y, folds, model, num_workers = 5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    with Pool(num_workers) as pool:
        pred_array = pool.map(worker_standart, args_list)

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

#uses sequential feature selection to find best combination of n features
def select_features(X, y, model, n_features):
    names = list(X.columns)
    sfs = SequentialFeatureSelector(estimator=model,n_features_to_select= n_features, n_jobs=-1, direction='forward')
    sfs.feature_names_in_ = names
    sfs.fit(X, y)
    fnames = sfs.get_feature_names_out()
    Xselected = sfs.transform(X)
    return Xselected, fnames


#uses sequential feature selection to find best combination of features
def selsect_features_all(X, y, model):
    repeats = len(X)
    currbest =0
    for i in range(1,repeats):
        tempX, names=select_features(X, y.to_numpy().flatten(), model, i)
        print('currently testing combination of ', i , 'features')
        temp_best= np.mean(stratified_cross_fold_validator(pd.DataFrame(tempX),y,5,model))
        if currbest< temp_best:
            currbest = temp_best
            features = names
    return currbest, features

#def worker_feature_selector:

