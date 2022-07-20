import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.ensemble import GradientBoostingClassifier

from joblib import dump, load
import lightgbm as lg


def one_hot_encoding(df, feature):
    data = df.copy()
    onehot = OneHotEncoder()
    onehot.fit(data[[feature]])

    encoded = onehot.transform(data[[feature]])
    data[onehot.categories_[0]] = encoded.toarray()
    return data.drop(feature, axis=1)


df_train = pd.read_csv('Train_Dataset_190649F.csv')
df_test = pd.read_csv('Test_Dataset_190649F.csv')

df_train.intertiol_plan.replace('no', 0, inplace=True)
df_train.intertiol_plan.replace('yes', 1, inplace=True)

df_test.intertiol_plan.replace('no', 0, inplace=True)
df_test.intertiol_plan.replace('yes', 1, inplace=True)

df_train.voice_mail_plan.replace('no', 0, inplace=True)
df_train.voice_mail_plan.replace('yes', 1, inplace=True)

df_test.voice_mail_plan.replace('no', 0, inplace=True)
df_test.voice_mail_plan.replace('yes', 1, inplace=True)

df_train.Churn.replace('No', 0, inplace=True)
df_train.Churn.replace('Yes', 1, inplace=True)

df_train = one_hot_encoding(df_train, 'location_code')
df_test = one_hot_encoding(df_test, 'location_code')

# new features
df_train['total_charge'] = df_train['total_intl_charge'] + df_train['total_night_charge'] + df_train[
    'total_eve_charge'] + df_train['total_day_charge']
df_test['total_charge'] = df_test['total_intl_charge'] + df_test['total_night_charge'] + df_test['total_eve_charge'] + \
    df_test['total_day_charge']

df_train['total_calls'] = df_train['total_intl_calls'] + df_train['total_night_calls'] + df_train['total_eve_calls'] + \
    df_train['total_day_calls']
df_test['total_calls'] = df_test['total_intl_calls'] + df_test['total_night_calls'] + df_test['total_eve_calls'] + \
    df_test['total_day_calls']

df_train['total_min'] = df_train['total_intl_minutes'] + df_train['total_night_minutes'] + df_train['total_eve_min'] + \
    df_train['total_day_min']
df_test['total_min'] = df_test['total_intl_minutes'] + df_test['total_night_minutes'] + df_test['total_eve_min'] + \
    df_test['total_day_min']

df_train["no_of_plans"] = df_train['intertiol_plan'] + \
    df_train['voice_mail_plan']
df_test['no_of_plans'] = df_test['intertiol_plan'] + df_test['voice_mail_plan']

df_train['avg_call_mins'] = df_train['total_min'] / df_train['total_calls']
df_test['avg_call_mins'] = df_test['total_min'] / df_test['total_calls']

x_columns = ['account_length',
             'intertiol_plan',
             'voice_mail_plan',
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
             'customer_service_calls',
             445.0,
             452.0,
             547.0,
             'total_charge',
             'total_calls',
             'total_min',
             'no_of_plans',
             'avg_call_mins']

x = df_train[x_columns]
y = df_train['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)
# rf = RandomForestClassifier(random_state=1, n_estimators=400)
lgbm_model = lg.LGBMClassifier(
    random_state=0,
    n_estimators=435,
    num_leaves=35,
    max_depth=8,
    verbose=-1
)
lgbm_model.fit(x_train, y_train)

dump(lgbm_model, 'model.joblib')
print("model created.")
