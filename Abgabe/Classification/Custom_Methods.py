import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from itertools import permutations
from warnings import warn

from tqdm.auto import tqdm
import math

__all__ = ['stratified_cross_fold_validator', 'stratified_cross_fold_validator_for_smote', 'removeOutlier',
           'sequential_feature_selector', 'permutation_tester', 'sequential_feature_eliminator', 'stratified_cross_fold_validator_for_smote_single',
           'cross_fold_validator', 'read_data', 'cap_Outlier']





def worker_oversample(args):
    """
    this function is called by stratified_cross_fold_for_smote. it recives
    the features (X), the labels (y), train_index (indexes of the data used for training),
    test_index (indexes of the data used for testing) and the model to evaluate.
    This function essentaily gets one fold of cross-validation, perfmors upsampling
     on that fold, and returns the F1- score after appling SMOTE and fitting the specified data
    :param args:
    :return F1 score:
    """
    #get parameters from args
    X, y, train_index, test_index, model = args

    #SMOTE parameters
    smo = SMOTE(random_state=42)

    #Index of data to be used to fit the model is used to get the training data seperated from test data
    current_train_fold_y = y.iloc[train_index]
    current_train_fold_X = X.iloc[train_index]

    #Oversampling the training data using SMOTE
    current_train_fold_X, current_train_fold_y = smo.fit_resample(current_train_fold_X, current_train_fold_y)

    #Index of the data used to test the model is used to get the test data seperated from training data
    current_test_fold_y = y.iloc[test_index]
    current_test_fold_X = X.iloc[test_index]

    #fit the model with training data
    model.fit(current_train_fold_X, current_train_fold_y.to_numpy().flatten())

    #make predictions
    temp_pred = model.predict(current_test_fold_X)

    #calculate and return F1 macro score
    return f1_score(current_test_fold_y, temp_pred, average="macro")


def worker_standart(args):
    """
        this function is called by stratified_cross_fold. it recives
        the features (X), the labels (y), train_index (indexes of the data used for training),
        test_index (indexes of the data used for testing) and the model to evaluate.
        this function essentaily gets one fold of crossvalidation an returns its results
        :param args:
        :return : F1 score
        """

    #get parameters from args
    X, y, train_index, test_index, model = args

    # Index of data to be used to fit the model is used to get the training data seperated from test data
    current_train_fold_y = y.iloc[train_index]
    current_train_fold_X = X.iloc[train_index]

    #Index of the data used to test the model is used to get the test data seperated from training data
    current_test_fold_y = y.iloc[test_index]
    current_test_fold_X = X.iloc[test_index]

    #fit the model with training data
    model.fit(current_train_fold_X, current_train_fold_y.to_numpy().flatten())

    #make predictions
    temp_pred = model.predict(current_test_fold_X)

    #calculate and return F1 macro score
    return f1_score(current_test_fold_y, temp_pred, average="macro")


# needs pandas dataframe to work
def stratified_cross_fold_validator_for_smote(X, y, folds, model, num_workers=10):
    """
        this function usues Sci-kit Learn's StratifiedKFold to get k folds and calculate the scores of
        each fold. This version calls a worker which Performs SMOTE oversampling on each fold

    :param X: Pandas dataframe containign the features
    :param y: Pandas dataframe containing the labels
    :param folds:   Int, number of folds
    :param model:   untrained sklearn Model
    :param num_workers: Int number of workers. if workers > folds, workers = folds
    :return: numpy array with F1 macro scores of each fold
    """

    #get k folds
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    #parameters to pass to the workers
    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    #start n workers and wait for each worker to return its result
    with Pool(num_workers) as pool:
        scores = pool.map(worker_oversample, args_list)

    #pool cleanup
    pool.close()
    pool.join()

    #return numpy array with the results
    return np.array(scores, dtype=np.float32)

def stratified_cross_fold_validator_for_smote_single(X, y, folds, model):
    """
        this function usues Sci-kit Learn's StratifiedKFold to get k folds and calculate the scores of
        each fold. this version does not support multiple workers

    :param X: Pandas dataframe containign the features
    :param y: Pandas dataframe containing the labels
    :param folds:   Int, number of folds
    :param model:   untrained sklearn Model
    :return: numpy array with F1 macro scores of each fold
    """

    #get the n folds
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    #SMOTE parameters
    smo = SMOTE(random_state=42)

    #for now, empty array for scores
    scores=[]

    #loop that itterates over the train and test folds
    for train_index, test_index in skf.split(X, y):

        #Index of data to be used to fit the model is used to get the training data seperated from test data
        current_train_fold_y = y.iloc[train_index]
        current_train_fold_X = X.iloc[train_index]

        # Oversampling the training data using SMOTE
        current_train_fold_X, current_train_fold_y = smo.fit_resample(current_train_fold_X, current_train_fold_y)

        #Index of data to be used to test the model is used to get the training data seperated from test data
        current_test_fold_y = y.iloc[test_index]
        current_test_fold_X = X.iloc[test_index]

        #fit the model with current fold
        model.fit(current_train_fold_X, current_train_fold_y.to_numpy().flatten())

        #make predcictions on the current fold
        temp_pred = model.predict(current_test_fold_X)

        #add the score of the predcition of the current fold to the array
        scores.append(f1_score(current_test_fold_y, temp_pred, average="macro"))

    #return numpy array with scores
    return np.array(scores, dtype=np.float32)


def stratified_cross_fold_validator(X, y, folds, model, num_workers=10):
    """
        this function usues Sci-kit Learn's StratifiedKFold to get k folds and calculate the scores of
        each fold. this version does not perform smote

    :param X: Pandas dataframe containign the features
    :param y: Pandas dataframe containing the labels
    :param folds:   Int, number of folds
    :param model:   untrained sklearn Model
    :param num_workers: Int number of workers. if workers > folds, workers = folds
    :return: numpy array with F1 macro scores of each fold
    """
    # get the n folds
    skf = StratifiedKFold(n_splits=folds, shuffle=False)

    #parameters to pass to the workers
    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in skf.split(X, y)]

    #start n workers and wait for each worker to return its result
    with Pool(num_workers) as pool:
        scores = pool.map(worker_standart, args_list)

    #pool cleanup
    pool.close()
    pool.join()

    #return numpy array with scores
    return np.array(scores, dtype=np.float32)

def cross_fold_validator(X, y, folds, model, num_workers=10):
    """
    this function usues Sci-kit Learn's KFold to get k folds and calculate the scores of
        each fold. this version does not perform smote. it supports multiple workers

    :param X:
    :param y:
    :param folds:
    :param model:
    :param num_workers:
    :return:
    """
    #parameters for Kfold
    kf = KFold(n_splits=folds, shuffle=False)

    #parameters to pass to the workers
    args_list = [(X, y, train_index, test_index, model) for train_index, test_index in kf.split(X, y)]

    # start n workers and wait for each worker to return its result
    with Pool(num_workers) as pool:
        scores = pool.map(worker_standart, args_list)

    #pool cleanup
    pool.close()
    pool.join()

    #return scores in numpy array
    return np.array(scores, dtype=np.float32)

def removeOutlier(X, y, n):
    """
    this method removes outliers based on Sci-kit leanrs Local Outlier Factor
    :param X: pandas dataframe with features
    :param y: pandas dataframe with labels
    :return:
    """
    # outlier detection
    lof = LocalOutlierFactor(n_neighbors=n)

    #outliers are marked as -1
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

def cap_Outlier(X):
    """
    This FUnction caps outliers. The outliers are Detected by an Isolation Forest and capped according to the upper and
    lower bounds of the Inter Quartile Range
    :param X:
    :return:
    """
    pd.set_option("mode.copy_on_write", True)
    names = list(X.columns)

    # outlier detection
    lof = IsolationForest(n_jobs=-1,random_state=211, bootstrap=True, n_estimators= 700)
    outliers = lof.fit_predict(X)

    # convert -1 to 0 for boolean indexing
    outliers[outliers == -1] = 0

    bounds=[]

    for column in names:
        #Quartiles
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)

        #calculate IQR
        IQR = Q3 - Q1

        #define bounds
        lowerbound = Q1 - (1.5 * IQR)
        upperbound = Q3 + (1.5 * IQR)
        #print(lowerbound,upperbound)

        bounds.append((lowerbound,upperbound))

    #cap the outliers
    repeats = len(X)
    for i in range(repeats):
        if outliers[i] == 0:
            for j in range(len(list(X.columns))):
                low, up = bounds[j]
                if X.iloc[i,j] < low:
                    X.iloc[i,j] = low
                elif X.iloc[i,j] > up:
                    X.iloc[i,j] = up
    # return values
    return X



# needs Pandas dataframe to work
def sequential_feature_selector(X, y, model, verbose=False, remove_outlier = False):
    """
    Sequential feacture selector. does not use sklears oimplementation since sklearn only allows to get results for n
    features. and each next n nets to start over. this just does all. can remove outliers before testing
    :param X: pandas dataframe
    :param y:   pabdas dataframe
    :param model:   sklearn model
    :param verbose: Boolean
    :param remove_outlier: Boolean
    :return:
    """

    warn(
        'The method sequentail_feature_selector needs a lot of time by nature and it cant use multiple workers due to the usage of crossvalidation'
        'which uses multiple workers')
    if remove_outlier:
        X, y = removeOutlier(X,y)
    # number of features
    n_features = len(X.columns)

    # list with all (remaining) feature names
    names = list(X.columns)

    # this list collects the sequentialy best selected features
    selected_features = []

    # variables to remember best overall selection
    overall_best_features = []
    overall_best_score = 0

    #initilize the progressbar
    progress_bar = tqdm(total=n_features - 1, position=0, desc="Sequential Feature Selector",leave=True)


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
            tqdm.write(f'Score reached:  {curr_best_score} using: {selected_features}')

            #update the progressbar
            progress_bar.update(1)

    if verbose == True:
        tqdm.write(f'Best Selection: {overall_best_features} with score: {overall_best_score}')

    return overall_best_features


'# scine Gradoemt boosting is sensitive to the order of features the following method tests diferent permutainons of the supmittet dataframe'


def permutation_tester(X, y, model, verbose=False):
    """
    this function tests all possible permutations. not realy usefull but was used for a test

    :param X: pandas dataframe
    :param y: pandas dataframe
    :param model: skleanr model
    :param verbose: Boolean
    :return: string of combinations
    """
    warn('The Method permutation_terster does not support the use of multiple workers. it will take a very long time'
         'running it to he end is not recomended, instead set verbose True and let it run some time')

    #list of names of the features
    names = list(X.columns)
    #variables for the loop
    best_score = 0
    best_order = ''
    #number of permutations
    perms = permutations(names, len(names))

    #progressbar
    progress_bar = tqdm(total=math.factorial(len(X.columns)), position=0, desc="Permutation tester")

    #loop that itterates thorugh the permutatuions
    for perm in permutations(perms):
        perm = list(perm)
        temp = np.mean(stratified_cross_fold_validator(X[perm], y, 5, model))
        if best_score < temp:
            best_score = temp
            best_order = perm
            if verbose == True:
                print('new best order: ', best_order, 'with score: ', best_score)
    if verbose:
        progress_bar.update(1)
    return best_order




'#read the csv files'
def read_data(feature_path ='train_features.csv', label_path='train_label.csv'):
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    labels.drop(['Id'],axis=1,inplace=True)
    return features, labels

def sequential_feature_eliminator(X, y, model, verbose=False, remove_outlier =False):
    """
    Sequential feacture eliminator. does not use sklears oimplementation since sklearn only allows to get results for n
    features. and each next n nets to start over. this just does all. can remove outliers before testing
    :param X: pandas dataframe
    :param y:   pabdas dataframe
    :param model:   sklearn model
    :param verbose: Boolean
    :param remove_outlier: Boolean
    :return:
    """

    warn(
        'The method sequentail_feature_eliminator needs a lot of time by nature and it cant use multiple workers due to the usage of crossvalidation'
        'which uses multiple workers')
    if remove_outlier:
        X, y =removeOutlier(X,y)
    # number of features
    n_features = len(X.columns)

    # list with all (remaining) feature names
    names = list(X.columns)

    # this list collects the sequentialy best selected features
    selected_features = list(X.columns)

    # initialize the tqdm progress bar with position=0
    progress_bar = tqdm(total=n_features - 1, position=0, desc="Sequential Feature Elimination")

    # variables to remember best overall selection
    overall_best_features = []
    overall_best_score = 0

    for i in range(len(X.columns)):
        print(i)
        curr_best_score = 0
        curr_worst_feature = ''

        # this loop takes the so far selected features and tries all other features on it to select the next best one
        for j in range(len(names)):
            # take current selected features and not yet selected feature to try
            curr_feature_list = selected_features.copy()
            curr_feature_list.remove(names[j])
            temp_score = np.mean(stratified_cross_fold_validator(X[curr_feature_list], y, 5, model))

            if curr_best_score < temp_score:
                curr_best_score = temp_score
                curr_worst_feature = names[j]

                if overall_best_score < temp_score:
                    overall_best_features = curr_feature_list
                    overall_best_score = temp_score

        # remove the next worst feature
        selected_features.remove(curr_worst_feature)

        # remove the removed feature so it doesn't get removed multiple times resulting in an error
        names.remove(curr_worst_feature)

        if verbose:
            tqdm.write(f'Score reached: {curr_best_score} using: {selected_features}')

            # update the tqdm progress bar
            progress_bar.update(1)

    if verbose:
        tqdm.write(f'Best Selection: {overall_best_features} with score: {overall_best_score}')

    return overall_best_features


