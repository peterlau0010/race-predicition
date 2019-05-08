import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import logging
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
pd.set_option('mode.chained_assignment', None)
class Regression:

    logging.basicConfig(filename='WebCrawling.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

    def __init__(self, data):
        self.data = data
        self.originaldata = data

    def removeEmpty(self):
        # Group category
        self.data.loc[self.data['class'].str.contains(
            'Class 2'), 'class'] = 'Class 2'
        self.data.loc[self.data['class'].str.contains(
            'Class 3'), 'class'] = 'Class 3'
        self.data.loc[self.data['class'].str.contains(
            'Class 4'), 'class'] = 'Class 4'
        self.data.loc[self.data['road'].str.contains(
            '"A+'), 'road'] = 'TURF - "A" Course'
        self.data.loc[self.data['road'].str.contains(
            '"B+'), 'road'] = 'TURF - "B" Course'
        self.data.loc[self.data['road'].str.contains(
            '"C+'), 'road'] = 'TURF - "C" Course'

        # Remove '---' value
        self.data = self.data[self.data.finishTime != '---']
        # print(self.data.info())

        # Covert finishTime from string to float
        self.data['finishTime'] = (pd.to_datetime(
            self.data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)

        # Remove used col
        self.data = self.data.drop(['date', 'raceNo', 'raceName',
                                    'plc', 'horseNo', 'lbw', 'runPos', 'odds', 'horseName', 'money', 'code'], axis=1)

        # Process Category data
        self.data = pd.get_dummies(
            self.data, columns=['raceCource'], prefix=['raceCource'])
        self.data = pd.get_dummies(
            self.data, columns=['going'], prefix=['going'])
        self.data = pd.get_dummies(self.data, columns=[
            'draw'], prefix=['draw'])
        self.data = pd.get_dummies(
            self.data, columns=['jockey'], prefix=['jockey'])
        self.data = pd.get_dummies(
            self.data, columns=['trainer'], prefix=['trainer'])
        self.data = pd.get_dummies(self.data, columns=[
            'road'], prefix=['road'])
        self.data = pd.get_dummies(self.data, columns=[
            'dist'], prefix=['dist'])
        self.data = pd.get_dummies(self.data, columns=[
            'class'], prefix=['class'])

        # self.data.info()

        headers = ','.join(map(str, self.data.columns.values))
        np.savetxt('processedData.csv', self.data,
                   delimiter=',', fmt='%s', header=headers)

        return self.data

    def standardizeData(self):
        headers = ','.join(map(str, self.data.columns.values))

        self.data = preprocessing.scale(self.data)

        np.savetxt('processedData.csv', self.data,
                   delimiter=',', fmt='%s', header=headers)
        return self.data


raceCource = 'HV'
classes = 'Class 4'
dist = '1200M'
road = 'TURF - A Course'

# ========= Read CSV file =========
data = pd.read_csv('history_adv15161718.csv', header=0)
data = data.iloc[::-1]
logging.info('Original CSV Size, %s', str(np.shape(data)))


# ========= Group data =========
data.loc[data['class'].str.contains('Class 2'), 'class'] = 'Class 2'
data.loc[data['class'].str.contains('Class 3'), 'class'] = 'Class 3'
data.loc[data['class'].str.contains('Class 4'), 'class'] = 'Class 4'
data = data[(data['class'] == 'Class 2') | (
    data['class'] == 'Class 3') | (data['class'] == 'Class 4') | (data['class'] == 'Class 5') | (data['class'] == 'Class 1')]
data.loc[data['road'].str.contains('"A+'), 'road'] = 'TURF - "A" Course'
data.loc[data['road'].str.contains('"B+'), 'road'] = 'TURF - "B" Course'
data.loc[data['road'].str.contains('"C+'), 'road'] = 'TURF - "C" Course'
data.loc[data['plc'].str.contains('3 DH'), 'plc'] = '3'
data.loc[data['plc'].str.contains('2 DH'), 'plc'] = '2'
data.loc[data['plc'].str.contains('1 DH'), 'plc'] = '1'
data.loc[data['plc'].str.contains('4 DH'), 'plc'] = '4'
data.loc[data['plc'].str.contains('5 DH'), 'plc'] = '5'
data.loc[data['plc'].str.contains('6 DH'), 'plc'] = '6'
data.loc[data['plc'].str.contains('7 DH'), 'plc'] = '7'
data.loc[data['plc'].str.contains('8 DH'), 'plc'] = '8'
data.loc[data['plc'].str.contains('9 DH'), 'plc'] = '9'
data.loc[data['plc'].str.contains('10 DH'), 'plc'] = '10'
# data.loc[data['trainer']=='J Moore' ,'trainerrank'] = 1

# ========= Remove '---' value in finishTime =========
data = data[data.finishTime != '---']
data = data[data.plc != 'DISQ']


# ========= Convert finishTime from String to float =========
data['finishTime'] = (pd.to_datetime(
    data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)
data['plc'] = data['plc'].astype(float)

# ========= Remove Useless data =========
data = data[(data['dist'] == dist)]
data = data[(data['road']==road)]
data = data[(data['class']==classes)]
data = data[(data['raceCource']==raceCource)]


# ========= Remove outlier data =========
q = data["finishTime"].quantile(0.99)
data = data[data["finishTime"] < q]


data_original = data.copy()

data = data[['dist', 'draw', 'finishTime',
             'going', 'class', '# # Age','Win_y','Win_x']]

logging.info('Selected CSV Size, %s', str(np.shape(data)))


# ========= Convert Categories to 0 1 =========
data = pd.get_dummies(data, columns=[
    'dist'], prefix=['dist'])

# data = pd.get_dummies(data, columns=[
#     'road'], prefix=['road'])

data = pd.get_dummies(
    data, columns=['going'], prefix=['going'])

data = pd.get_dummies(
    data, columns=['class'], prefix=['class'])
# data = pd.get_dummies(
# data, columns=['Country Of Origin'], prefix=['Country Of Origin'])
# data = pd.get_dummies(
#     data, columns=['Sire'], prefix=['Sire'])
# data = pd.get_dummies(
#     data, columns=['Dam'], prefix=['Dam'])
# data = pd.get_dummies(
#     data, columns=['jockey'], prefix=['jockey'])
# data = pd.get_dummies(
#     data, columns=['trainer'], prefix=['trainer'])
logging.info('After converted categories Size, %s', str(np.shape(data)))
# logging.info('\n {}'.format(data.head()))

data = data.astype(float)

# ========= Prepare X Y =========
X = data.drop(['finishTime'], axis=1)
y = data[['finishTime']]

# ========= split train and test =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False)

# ========= Standardization for data =========
scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
# y_train = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
# y_test = scalery.transform(y_test)

# X_train = scalerX.fit_transform(X_train)
# X_test = scalery.fit_transform(X_test)


# ========= Regression Model =========
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# ========= Prediction =========
y_pred = model.predict(X_test)


# y_pred = scalery.inverse_transform(y_pred)
logging.info(y_pred[:5])
logging.info(y_test[:5])

X_test = scalerX.inverse_transform(X_test)

# ========= Prepare output csv ==========
y_test.loc[:,'y_pred'] = y_pred
# print(y_test)
df_out = pd.merge(data_original, y_test, how='left',
                  left_index=True, right_index=True)

df_out = df_out[(df_out['y_pred'] > 0) & (df_out['y_pred'] < 200000)]
df_out["y_Rank"] = df_out.groupby(['date', 'raceNo'])["y_pred"].rank()
df_out["x_Rank"] = df_out.groupby(['date', 'raceNo'])["plc"].rank()

df_out = df_out.sort_values(
    ['date', 'raceNo', 'plc'], ascending=[True, False, True])

df_out = df_out[(df_out['y_Rank'] <= 1)]

regression_report = df_out[['# # Age', 'Chinese Name', 'Code', 'finishTime_y', '# Trainer', '# Jockey',
                            'plc', 'odds', 'finishTime_x', 'plc', 'odds', 'finishTime_x', 'finishTime_y', 'y_pred', 'y_Rank', 'x_Rank']]

headers = ','.join(map(str, regression_report.columns.values))
np.savetxt('regression_report.csv', regression_report.round(0),
           delimiter=',', fmt='%s', header=headers)

logging.info(np.shape(df_out))
df_out = df_out[df_out.plc.notnull()]
df_out = df_out[df_out.y_Rank.notnull()]
logging.info(np.shape(df_out))


# The coefficients
logging.info('Coefficients: \n %s', model.coef_)
# The mean squared error
logging.info("Mean squared error: %.2f"
             % mean_squared_error(df_out['x_Rank'], df_out['y_Rank']))
# Explained variance score: 1 is perfect prediction
logging.info('Variance score: %.2f' %
             r2_score(df_out['x_Rank'], df_out['y_Rank']))

