import pandas as pd
import numpy as np
import os.path
import pickle

data_path = ['..', 'data']


def MAE(model, patientID, X, y, n_splits=5):
    '''
    Returns the mean of a number (five by default) mae's for the method.
        model: must have the methods fit(X,y) and predict(X,y) to work
        patientID: to stratify train/test splits
        X: endogenous variables or predictors
        y: exogenous variable or outcome
        n_splits: the number of times to find the mae

    Example 1
    >>>data = pd.read_csv("master_frame5.csv").dropna()
    >>>X = np.ones(data.shape[0]).reshape(-1, 1)
    >>>print(MAE(LinearRegression(), patientID=patientID, X=X, y=data['RawScore']))
    ...0.03428465274515196
    '''
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error

    if len(X.shape) == 1:
        X = np.array(X).reshape(-1, 1)
    else:
        X = np.array(X)
    y = np.array(y).ravel()
    
    patientID.index = range(patientID.shape[0])
    patient_list = patientID.unique()
    kf = KFold(n_splits=5, shuffle=True)
    mae = []
    for train_p, test_p in kf.split(patient_list):
        train_index = patientID[patientID.isin(patient_list[train_p])].index
        test_index = patientID[patientID.isin(patient_list[test_p])].index
        model.fit(X[train_index], y[train_index])
        mae.append(
            mean_absolute_error(y[test_index], model.predict(X[test_index]))
        )
    return np.mean(mae)



def METRIC(model, metric, patientID, X, y, n_splits=5):
    '''
    A generic version of MAE.
        metric: A function that computes a metric like so, metric(y, y_hat)
    '''
    from sklearn.model_selection import KFold

    if len(X.shape) == 1:
        X = np.array(X).reshape(-1, 1)
    else:
        X = np.array(X)
    y = np.array(y).ravel()
    
    patientID.index = range(patientID.shape[0])
    patient_list = patientID.unique()
    kf = KFold(n_splits=5, shuffle=True)
    m = []
    for train_p, test_p in kf.split(patient_list):
        train_index = patientID[patientID.isin(patient_list[train_p])].index
        test_index = patientID[patientID.isin(patient_list[test_p])].index
        test_index = patientID.isin(patient_list[test_p]).index
        model.fit(X[train_index], y[train_index])
        m.append(
            metric(y[test_index], model.predict(X[test_index]))
        )
    return np.mean(m)


def GetPatientXy(X_columns=None, y_column=None, filter=None, dropna=True, csv_file='master_frame5.csv'):
    '''
    X_columns: used tp specify the columns returned in X, otherwise those
        marked as X_column in Columns.csv are returned.
    y_column: used to specify the outcome column. If nothing is passed, returns
        all columns marked as y_column in Columns.csv.
    filter: used to filter data before returning X and y. Pass a dictionary
        where the key is the column and the value is a string of the
        condition whereby the column will be filtered.
    dropna: drops rows that have NAs in master_frame5.csv. These would be
        sessions that have fewer than two OQ scores

    Examples:
    patientID, X, y = GetPatientXy(dropna=False)
    patientID, X, y = GetPatientXy(y_column='NetDrop')
    patientID, X, y = GetPatientXy(filter={'NumberOfOQs': ' >= 5',
                         'IncomingSubClinical': ' != 1'})
    '''
    if os.path.exists(os.path.join(*data_path, csv_file)) == False:
        print("The csv file does not exist")
        return None

    data = pd.read_csv(os.path.join(*data_path, csv_file))
    columns = pd.read_csv(os.path.join(*data_path, 'DataDictionary.csv'))
    if filter is not None:
        for key in filter.keys():
            mask = eval('data["{}"]{}'.format(key, filter[key]))
            data = data.loc[mask, :]
    if dropna:
        data = data.dropna()
    if X_columns is None:
        X_columns = columns.loc[columns['Type'] == 'X_column', 'Name'].values
    X = data[X_columns]
    if 'TherapistID' in X_columns:
        X = pd.get_dummies(X, columns=['TherapistID'], drop_first=False)
    if y_column is None:
        y_column = columns.loc[columns['Type'] == 'y_column', 'Name'].values
    y = data[y_column]
    return (data['PatientID'], X, y)


def GetCurrentModel(pickle_file_name="current_model.pkl"):
    with open(pickle_file_name, 'rb') as f:
        model = pickle.load(f)
    return model


def GetFeaturesList(file_name="features.txt"):
    features_list = []
    with open(file_name, 'r') as f:
        for line in f:
            features_list.append(line.strip())

    #Check to see the features list was read in properly
    if (len(features_list) == 0):
        print("Something went wrong in reading the features file")
    return features_list


def GetModelPatientXy(input = ("master_frame5.csv", "features.txt", "NetDrop", "current_model.pkl")):
    model = GetCurrentModel(input[3])
    features_list = GetFeaturesList(file_name=input[1])
    patientID, X,y = GetPatientXy(X_columns = features_list, y_column = input[2], dropna = True, csv_file = input[0])
    return (model, patientID, X, y)


def ProgressBar(index, total):
    '''
    This function uses values 0 through 100 to create a progress bar.
    Make sure the index will reach the total. You may, for example, need
    to subract one from total.
    '''
    percent_no_round = index / total * 100

    def print_bar(percent):
        p = round(percent / 5)
        if percent < 100 and percent >= 0:
            return print('\r[', '\033[0;43m', ' ' * p, '\033[0m', ' ' * (20 - p), ']', percent, '%', end='', sep='', flush=True)
        if percent == 100:
            return print('\r[', '\033[0;42m', ' ' * 20, '\033[0m', ']', percent, '%', end='\n', sep='', flush=True)

    if total < 1000:
        if abs(percent_no_round % 1) <= .1:
            print_bar(round(percent_no_round))
    else:
        if (percent_no_round % 1) > (((index + 1) / total * 100) % 1):
            print_bar(round(percent_no_round))
