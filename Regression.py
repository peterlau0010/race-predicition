import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time

dist = '1200M'
going = 'GOOD'

# Read CSV file
data = pd.read_csv('history_adv.csv', header=0)
data = data.iloc[::-1]
print(np.shape(data))

# Group category
data.loc[data['class'].str.contains('Class 2'), 'class'] = 'Class 2'
data.loc[data['class'].str.contains('Class 3'), 'class'] = 'Class 3'
data.loc[data['class'].str.contains('Class 4'), 'class'] = 'Class 4'
data.loc[data['road'].str.contains('"A+'), 'road'] = 'TURF - "A" Course'
data.loc[data['road'].str.contains('"B+'), 'road'] = 'TURF - "B" Course'
data.loc[data['road'].str.contains('"C+'), 'road'] = 'TURF - "C" Course'

# Remove '---' value
data = data[data.finishTime != '---']
data = data[data.awt != '---']
data = data[data.dhw != '---']
data = data[(data['class'] == 'Class 2') | (
    data['class'] == 'Class 3') | (data['class'] == 'Class 4') | (data['class'] == 'Class 5') | (data['class'] == 'Class 1')]

data = data[(data['trainer'] != 'J M Moore') & (
    data['trainer'] != 'M C Tam') & (data['trainer'] != 'P Leyshan') & (data['trainer'] != 'S H Cheong') & (data['trainer'] != 'W H Tse') & (data['trainer'] != 'Y C Fung') & (data['trainer'] != 'K C Chong')]

jockeylist = data['jockey'].value_counts()
jockeylist = jockeylist[jockeylist > 300]
data = data[data['jockey'].isin(jockeylist.index.values.tolist())]

trainerlist = data['trainer'].value_counts()
trainerlist = trainerlist[trainerlist > 30]
data = data[data['trainer'].isin(trainerlist.index.values.tolist())]

Damlist = data['Dam'].value_counts()
Damlist = Damlist[Damlist > 5]
data = data[data['Dam'].isin(Damlist.index.values.tolist())]

Sirelist = data['Sire'].value_counts()
Sirelist = Sirelist[Sirelist > 5]
data = data[data['Sire'].isin(Sirelist.index.values.tolist())]


# print(data.head())
# data.groupby('trainer')['horseName'].nunique().plot(kind='bar')
# plt.show()
# exit()


# Match information
# data = data[data.dist == dist]
# data = data[data.going == going]


# Prepare reformat data
reformat_data = data.copy()

# Covert finishTime from string to float
reformat_data['finishTime'] = (pd.to_datetime(
    reformat_data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)

X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    reformat_data, reformat_data, test_size=0.10)

# Process Category data
reformat_data = pd.get_dummies(
    reformat_data, columns=['raceCource'], prefix=['raceCource'])
reformat_data = pd.get_dummies(
    reformat_data, columns=['going'], prefix=['going'])
reformat_data = pd.get_dummies(
    reformat_data, columns=['jockey'], prefix=['jockey'])
reformat_data = pd.get_dummies(
    reformat_data, columns=['trainer'], prefix=['trainer'])
reformat_data = pd.get_dummies(reformat_data, columns=[
    'draw'], prefix=['draw'])
reformat_data = pd.get_dummies(
    reformat_data, columns=['class'], prefix=['class'])
# reformat_data = pd.get_dummies(reformat_data, columns=[
#     'money'], prefix=['money'])
reformat_data = pd.get_dummies(reformat_data, columns=[
    'road'], prefix=['road'])
reformat_data = pd.get_dummies(reformat_data, columns=[
                               'dist'], prefix=['dist'])
reformat_data = pd.get_dummies(reformat_data, columns=[
                               'Country Of Origin'], prefix=['Country Of Origin'])
reformat_data = pd.get_dummies(reformat_data, columns=[
                               'Dam'], prefix=['Dam'])
reformat_data = pd.get_dummies(reformat_data, columns=[
                               'Sire'], prefix=['Sire'])
reformat_data = pd.get_dummies(reformat_data, columns=[
                               'Import Type'], prefix=['Import Type'])

reformat_data = pd.get_dummies(reformat_data, columns=[
                               '# Age'], prefix=['# Age'])
# Split train and test
X = reformat_data.drop(['finishTime', 'date', 'raceNo', 'raceName',
                        'plc', 'horseNo', 'lbw', 'runPos', 'odds', 'horseName', 'money','Chinese Name','English Name','Code','Foaling Date','Former Name','_id','code' ], axis=1)
y = reformat_data['finishTime']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10)

print(X_train.columns.values)

# Create linear regression object
regr = linear_model.LinearRegression()

# Create RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)

# Train the model using the training sets
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)


# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs

plt.scatter(y_test, y_pred)
plt.ylim(68000, 73000)
plt.xlim(68000, 73000)
plt.show()

X_test_original['p_finishTime'] = y_pred

# print(X_test_original.head())
headers = ','.join(map(str, X_test_original.columns.values))

np.savetxt('regression_report.csv', X_test_original.round(0),
           delimiter=',', fmt='%s', header=headers)
