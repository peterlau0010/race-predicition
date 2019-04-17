import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import time
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model

data = pd.read_csv('history_bak.csv',header=0)
# print(data.head())

# print(data['raceCource'].unique())
# print(data['going'].unique())

data_onehot = data.copy()
print(data_onehot.columns.tolist())
data_onehot = pd.get_dummies(data_onehot, columns=['raceCource'], prefix = ['raceCource'])
data_onehot = pd.get_dummies(data_onehot, columns=['going'], prefix = ['going'])
data_onehot = pd.get_dummies(data_onehot, columns=['jockey'], prefix = ['jockey'])
data_onehot = pd.get_dummies(data_onehot, columns=['trainer'], prefix = ['trainer'])
data_onehot = pd.get_dummies(data_onehot, columns=['draw'], prefix = ['draw'])
data_onehot = pd.get_dummies(data_onehot, columns=['class'], prefix = ['class'])
data_onehot = pd.get_dummies(data_onehot, columns=['money'], prefix = ['money'])
data_onehot = pd.get_dummies(data_onehot, columns=['road'], prefix = ['road'])
data_onehot = pd.get_dummies(data_onehot, columns=['dist'], prefix = ['dist'])

#Remove '---' value
data_onehot = data_onehot[data_onehot.finishTime != '---']

#Move finishTime to last column
cols = data_onehot.columns.tolist()
cols = cols[0:10]  + cols[11:]+ cols[10:11]
data_onehot = data_onehot[cols]

#Drop useless columns
data_onehot = data_onehot.drop(['date', 'raceNo','raceName','plc','horseNo','lbw','runPos','odds','horseName'], axis=1)

# print(data_onehot.columns.tolist())
data_onehot['finishTime'] = (pd.to_datetime(data_onehot['finishTime'],format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)
print(data_onehot.head())


X = data_onehot.drop(['finishTime'], axis=1)
y = data_onehot['finishTime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)


print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training set
regr.fit(X_train, y_train)

print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)


# Make predictions using the testing set
y_pred = regr.predict(X_test)

print(type(y_test))
print(type(X_test))
print(type(y_pred))

X_test['finishTime'] = y_test.values
X_test['predicitTime'] = y_pred.toarray().tolist()
print(y_test.head())


