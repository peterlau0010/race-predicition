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

from joblib import dump, load

pd.set_option('mode.chained_assignment', None)


class Regression:

    logging.basicConfig(filename='./Log/Regression.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

    def __init__(self, data, raceCourse, classes, dist, road):
        self.data = data
        self.originaldata = data
        self.raceCourse = raceCourse
        self.classes = classes
        self.dist = dist
        self.road = road

    def groupData(self, data):
        # ========= Group data =========
        data.loc[data['class'].str.contains('Class 2'), 'class'] = 'Class 2'
        data.loc[data['class'].str.contains('Class 3'), 'class'] = 'Class 3'
        data.loc[data['class'].str.contains('Class 4'), 'class'] = 'Class 4'
        data = data[(data['class'] == 'Class 2') | (
            data['class'] == 'Class 3') | (data['class'] == 'Class 4') | (data['class'] == 'Class 5') | (data['class'] == 'Class 1')]
        data.loc[data['road'].str.contains('"A+'), 'road'] = 'TURF - A Course'
        data.loc[data['road'].str.contains('"B+'), 'road'] = 'TURF - B Course'
        data.loc[data['road'].str.contains('"C+'), 'road'] = 'TURF - C Course'
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

        return data

    def removeInvalidData(self, data):
        # ========= Remove '---' value in finishTime =========
        data = data[data.finishTime != '---']
        data = data[data.plc != 'DISQ']

        return data

    def convertData(self, data):
        # ========= Convert finishTime from String to float =========
        data['finishTime'] = (pd.to_datetime(
            data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)
        data['plc'] = data['plc'].astype(float)

        return data

    def selectAppropriateData(self, data):
        data = data if self.dist is None else data[(data['dist'] == self.dist)]
        data = data if self.road is None else data[(data['road'] == self.road)]
        data = data if self.classes is None else data[(
            data['class'] == self.classes)]
        data = data if self.raceCourse is None else data[(
            data['raceCourse'] == self.raceCourse)]
        return data

    def removeOutlier(self, data):
        q = data["finishTime"].quantile(0.99)
        data = data[data["finishTime"] < q]
        return data


raceCourse, classes, dist, road = None, None, None, None

# raceCourse = 'HV'
# classes = 'Class 4'
# dist = '1200M'
# road = 'TURF - A Course'

# ========= Read CSV file =========
data = pd.read_csv('./Processed Data/history_csv_merged.csv', header=0)
data = data.iloc[::-1]
logging.info('Original CSV Size, %s', str(np.shape(data)))

# ======== initial Regression Class ============
r = Regression(data, raceCourse, classes, dist, road)

# ======== Prepare Data ==============
data = r.groupData(data)

data = r.removeInvalidData(data)
logging.info('After removeInvalidData Size, %s', str(np.shape(data)))

data = r.convertData(data)

data = r.selectAppropriateData(data)
logging.info('After selectAppropriateData Size, %s', str(np.shape(data)))

data = r.removeOutlier(data)
logging.info('After removeOutlier Size, %s', str(np.shape(data)))


# ========= Select column for regresion ============

data_original = data.copy()

data = data[['road', 'dist', 'class', 'draw', 'finishTime',
             'going', 'Age', 'Win%_y', 'Win%_x', 'DamRank', 'SireRank', 'awt', 'dhw', 'raceCourse']]

logging.info('Selected CSV Size, %s', str(np.shape(data)))


# ========= Convert Categories to 0 1 =========
data = pd.get_dummies(data, columns=[
    'dist'], prefix=['dist'])

data = pd.get_dummies(data, columns=[
    'road'], prefix=['road'])

data = pd.get_dummies(
    data, columns=['going'], prefix=['going'])

data = pd.get_dummies(
    data, columns=['class'], prefix=['class'])

data = pd.get_dummies(
    data, columns=['raceCourse'], prefix=['raceCourse'])

logging.info('After converted categories Size, %s', str(np.shape(data)))

data = data.astype(float)
data.fillna(data.mean(), inplace=True)


# ========= Prepare X Y =========
X = data.drop(['finishTime'], axis=1)
headers = ','.join(map(str, X.columns.values))


y = data[['finishTime']]

# ========= split train and test =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False)

np.savetxt('./Report/predicitValue.csv', X_test.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')
# ========= Standardization for data =========
scalerX = StandardScaler().fit(X_train)
dump(scalerX, 'scaler.sav')
# scalery = StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
# y_train = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
# y_test = scalery.transform(y_test)

# X_train = scalerX.fit_transform(X_train)
# X_test = scalery.fit_transform(X_test)


# ========= Regression Model =========

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
dump(model, 'regressioin_model.sav')

# ========= Prediction =========
y_pred = model.predict(X_test)


X_test = scalerX.inverse_transform(X_test)

# ========= Prepare output csv ==========
y_test.loc[:, 'y_pred'] = y_pred
# print(y_test)
df_out = pd.merge(data_original, y_test, how='left',
                  left_index=True, right_index=True)

df_out = df_out[(df_out['y_pred'] > 0) & (df_out['y_pred'] < 200000)]
df_out["y_Rank"] = df_out.groupby(['date', 'raceNo'])["y_pred"].rank()
df_out["x_Rank"] = df_out.groupby(['date', 'raceNo'])["plc"].rank()

df_out = df_out.sort_values(
    ['date', 'raceNo', 'plc'], ascending=[True, False, True])

# df_out = df_out[(df_out['y_Rank'] <= 1)]

regression_report = df_out[['Age', 'draw','Chinese Name', 'Code', 'Trainer', 'Jockey',
                            'plc', 'odds', 'finishTime_x', 'y_pred', 'y_Rank', 'x_Rank']]

headers = ','.join(map(str, regression_report.columns.values))
np.savetxt('./Report/regression_report.csv', regression_report.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')

logging.info(np.shape(df_out))
df_out = df_out[df_out.plc.notnull()]
df_out = df_out[df_out.y_Rank.notnull()]
logging.info(np.shape(df_out))

logging.info(X.columns.values)
# The coefficients
logging.info('Coefficients: \n %s', model.coef_)
for idx, col_name in enumerate(X.columns):
    logging.info("The coefficient for {} is {}".format(
        col_name, model.coef_[0][idx]))
# The mean squared error
logging.info("Mean squared error: %.2f"
             % mean_squared_error(df_out['x_Rank'], df_out['y_Rank']))
# Explained variance score: 1 is perfect prediction
logging.info('Variance score: %.2f' %
             r2_score(df_out['x_Rank'], df_out['y_Rank']))
