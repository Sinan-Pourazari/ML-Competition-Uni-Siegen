import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from itertools import permutations
from warnings import warn

__all__ = ['stratified_cross_fold_validator', 'stratified_cross_fold_validator_for_smote', 'removeOutlier',
           'sequential_feature_selector'
    , 'permutation_tester']


# needs pandas dataframe to work


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


# needs pandas dataframe to work
def stratified_cross_fold_validator_for_smote(X, y, folds, model, num_workers=10):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    with Pool(num_workers) as pool:
        scores = pool.map(worker_oversample, args_list)
    pool.close()
    pool.join()

    return np.array(scores, dtype=np.float32)


def stratified_cross_fold_validator(X, y, folds, model, num_workers=10):
    skf = StratifiedKFold(n_splits=folds, shuffle=False)
    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    with Pool(num_workers) as pool:
        scores = pool.map(worker_standart, args_list)
    pool.close()
    pool.join()

    return np.array(scores, dtype=np.float32)


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


# needs Pandas dataframe to work
def sequential_feature_selector(X, y, model, verbose=False):
    warn(
        'The method sequentail_feature_selector needs a lot of time by nature and it cant use multiple workers due to the usage of crossvalidation'
        'which uses multiple workers')
    # number of features
    n_features = len(X.columns)

    # list with all (remaining) feature names
    names = list(X.columns)

    # this list collects the sequentialy best selected features
    selected_features = []

    # variables to remember best overall selection
    overall_best_features = []
    overall_best_score = 0

    # this loop is used to ge the next best feature
    for i in range(n_features):
        curr_best_score = 0
        curr_best_feature = ''

        # this loop takes the so far selected features and tries all other features on it to select the next best one
        for j in range(len(names)):
            '#take current selected features and not yet selected feature to try'
            curr_feature_list = selected_features + [names[j]]
            temp_score = np.mean(stratified_cross_fold_validator(X[curr_feature_list], y, 5, model))
            if curr_best_score < temp_score:
                curr_best_score = temp_score
                curr_best_feature = names[j]

                if overall_best_score < temp_score:
                    overall_best_features = curr_feature_list
                    overall_best_score = temp_score

        '# append next best feature'
        selected_features.append(curr_best_feature)

        '#remove seltected feature so it doesnt get selected multiple times'
        names.remove(curr_best_feature)
        if verbose == True:
            print('Score reached: ', curr_best_score, 'using: ', selected_features)

    if verbose == True:
        print('best Selection: ', overall_best_features, 'with score: ', overall_best_score)

    return overall_best_features


'# scine Gradoemt boosting is sensitive to the order of features the following method tests diferent permutainons of the supmittet dataframe'


def permutation_tester(X, y, model, verbose=False):
    warn('The Method permutation_terster does not support the use of multiple workers. it might take some time')
    names = list(X.columns)
    best_score = 0
    best_order = ''
    for perm in permutations(names):
        perm = list(perm)
        temp = np.mean(stratified_cross_fold_validator(X[perm], y, 5, model))
        if best_score < temp:
            best_score = temp
            best_order = perm
            if verbose == True:
                print('new best order: ', best_order, 'with score: ', best_score)
    return best_order




'#read the csv files'
def read_data(feature_path='train_features.csv', label_path='train_label.csv'):
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    return features, labels
