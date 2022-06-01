# libs
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def drop_columns(df, columns):
    df.drop(columns, axis=1, inplace=True, errors='ignore')


def drop_duplicates(df):
    df_train.drop_duplicates(inplace=True)


def mark_minus_invalids_as_nan(df, columns):
    for col in columns:
        # df[df[col]<0][col]=np.nan
        df.loc[df[col] < 0, col] = np.nan


def set_nan(df, bounds):
    data = df.copy()
    for key in bounds.keys():
        # print("values",bounds[key])
        data.loc[data[key] < bounds[key][0], key] = np.nan
        data.loc[data[key] > bounds[key][1], key] = np.nan
    return data


def isNaN(num):
    if float('-inf') < float(num) < float('inf'):
        return False
    else:
        return True


def categorical_features_imputation_by_mode(df, features):
    for f in features:
        mode = df_train[~df_train[f].isnull()][f].value_counts().idxmax()
        # print(mode)
        df[f] = df[f].fillna(mode)
    return df


def regression_imputation(df, sets):
    for s in sets:
        # print(s)
        df.loc[df[s[0]].isnull() & df[s[1]].isnull(), s[0]] = df[s[0]].median()
        df.loc[df[s[0]].isnull() & df[s[1]].isnull(), s[1]] = df[s[1]].median()

        # print(df_test.iloc[[357]])
        # return
        lr_0 = LinearRegression()
        lr_1 = LinearRegression()
        valid = df.dropna(subset=s)
        # train_set_0 =df.dropna(subset=[s[0]])[s]
        # train_set_1 =df.dropna(subset=[s[1]])[s]
        # return valid

        a = valid[[s[0]]]
        b = valid[[s[1]]]
        # print(a)
        lr_0.fit(a, b)
        lr_1.fit(b, a)
        #
        #
        #
        for i in df.index:
            if isNaN(df[s[0]][i]):
                v = np.array([[df[s[1]][i]]])
                # print(i,df[s[0]][i],v)
                # return
                df[s[0]][i] = lr_1.predict(v)[0][0]

            elif isNaN(df[s[1]][i]):
                v = np.array([[df[s[0]][i]]])

                df[s[1]][i] = lr_0.predict(v)[0][0]

    return df
    # df.loc[df[s[1]].isnull(),s[1]] = lr_0.predict(df[[s[0]]])[df[s[1]].isnull()]
    # df.loc[df[s[0]].isnull(),s[0]] = lr_1.predict(df[[s[1]]])[df[s[0]].isnull()]


def handle_missing_by_median(df, features):
    data = df.copy()
    for feature in features:
        data.loc[data[feature].isnull(), feature] = df[feature].median()
    return data


# import dataset
df_train = pd.read_csv('Train_Dataset.csv')
df_test = pd.read_csv('Test_Dataset.csv')

# baseline for IDs
df_baseline_train = df_train.copy()
df_baseline_test = df_test.copy()

drop_features = ['customer_id', 'Unnamed: 19', 'Unnamed: 20']
drop_columns(df_train, drop_features)
drop_columns(df_test, drop_features)

drop_duplicates(df_train)
drop_duplicates(df_test)

# df_train.dropna(inplace=True, subset=['Churn'])

train_categorical = ['intertiol_plan', 'voice_mail_plan', 'Churn', 'location_code']
test_categorical = ['intertiol_plan', 'voice_mail_plan', 'location_code']
numerical = ['account_length',
             'number_vm_messages',
             'total_day_min',
             'total_day_calls',
             'total_day_charge',
             'total_eve_min',
             'total_eve_calls',
             'total_eve_charge',
             'total_night_minutes',
             'total_night_calls',
             'total_night_charge',
             'total_intl_minutes',
             'total_intl_calls',
             'total_intl_charge',
             'customer_service_calls']

mark_minus_invalids_as_nan(df_train, numerical)
mark_minus_invalids_as_nan(df_test, numerical)

suggested_bounds = {'account_length': [0, 250],
                    'number_vm_messages': [0, 51],
                    'total_day_min': [0, 500],
                    'total_day_calls': [0, 800],
                    'total_day_charge': [0, 61],
                    'total_eve_min': [0, 800],
                    'total_eve_calls': [0, 170],
                    'total_eve_charge': [0, 31],
                    'total_night_minutes': [0, 800],
                    'total_night_calls': [0, 175],
                    'total_night_charge': [0, 200],
                    'total_intl_minutes': [0, 25],
                    'total_intl_calls': [0, 20],
                    'total_intl_charge': [0, 10],
                    'customer_service_calls': [0, 10]}

df_train = set_nan(df_train, suggested_bounds)
df_test = set_nan(df_test, suggested_bounds)

correlated_sets = [['total_day_charge', 'total_day_min'],
                   ['total_eve_charge', 'total_eve_min'],
                   ['total_night_charge', 'total_night_minutes'],
                   ['total_intl_charge', 'total_intl_minutes']]

df_train = regression_imputation(df_train, correlated_sets)
df_test = regression_imputation(df_test, correlated_sets)

df_train = categorical_features_imputation_by_mode(df_train, train_categorical)
df_test = categorical_features_imputation_by_mode(df_test, test_categorical)

df_train = handle_missing_by_median(df_train, numerical)
df_test = handle_missing_by_median(df_test, numerical)

dataset_train = pd.concat([df_baseline_train['customer_id'], df_train], axis=1).reindex(df_train.index)
dataset_test = pd.concat([df_baseline_test['customer_id'], df_test], axis=1).reindex(df_test.index)
# Write the pre-processed dataset into a csv file --------------------------------

student_id = "190649F.csv"

dataset_train.to_csv('Train_Dataset_' + student_id, index=False)
dataset_test.to_csv('Test_Dataset' + student_id, index=False)
