
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Raw Data/history_bak.csv', header=0, low_memory=False)
data['date'] = data['date'].str.replace('/', '').astype(float)

print(np.shape(data))

split_date = 20190101

X_train, X_test = data[data['date'] < split_date].drop(
    ['finishTime'], axis=1), data[data['date'] >= split_date].drop(['finishTime'], axis=1)
y_train, y_test = data[data['date'] < split_date][['finishTime']], data[data['date']
                                                             >= split_date][['finishTime']]
# X_train, temp, y_train, temp = train_test_split(
#     X1, y1, test_size=0, shuffle=False)

# temp, X_test, temp, y_test = train_test_split(
#     X2, y2, test_size=0.999999, shuffle=False)

# print(X_train.head())
# print(X_test.head())

# print(y_train.head())
# print(y_test.head())
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))