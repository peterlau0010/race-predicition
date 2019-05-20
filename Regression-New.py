import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --------- Setting
logging.basicConfig(filename='./Log/Regression-New.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# --------- Read CSV
data = pd.read_csv('Processed Data/history_csv_merged.csv', header=0)
data = data.iloc[::-1]
data = data[data.finishTime != '---']
data = data[data.plc != 'DISQ']
data['plc'] = data['plc'].str.replace('.DH', '', regex=True)
data['finishTime'] = (pd.to_datetime(
    data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)
data['plc'] = data['plc'].astype(float)
logging.info('Original data %s \n %s', np.shape(data), data.head())


# --------- Select data for prediction
data = data[(data['dist'] == '1200M') & (data['raceCourse'] == 'ST') & (data['road'].str.contains('TURF')) & (data['going'].str.contains('GOOD'))]
q = data["finishTime"].quantile(0.99)
data = data[data["finishTime"] < q]
data_original = data.copy() 
logging.info('Selected data %s \n %s', np.shape(data), data.head())
# logging.info('Selected data %s \n %s', np.shape(data['plc']), data['plc'].value_counts())


# --------- Prepare Train Test
X = data.drop(['finishTime'], axis=1)
X_copy = X.copy() 
y = data[['finishTime']]


X = X[[ 'draw','Age','awt','dhw','J_Win','T_Win','Sire','Dam','plc']]

# X = pd.get_dummies(X, columns=[
#     'Age'], prefix=['Age'])

# X = pd.get_dummies(X, columns=[
#     'draw'], prefix=['draw'])

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, shuffle=False)

sumOfPlc = data.groupby(['Sire'])['plc'].sum()
noOfMatch = data.groupby(['Sire'])['finishTime'].count()
sireRank = sumOfPlc/noOfMatch
sireRank = sireRank.reset_index()
sireRank.columns = ['Sire','SireRank']

X_train = pd.merge(X_train, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])

sumOfPlc = data.groupby(['Dam'])['plc'].sum()
noOfMatch = data.groupby(['Dam'])['finishTime'].count()
damRank = sumOfPlc/noOfMatch
damRank = damRank.reset_index()
damRank.columns = ['Dam','DamRank']

X_train = pd.merge(X_train, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])

X_train = X_train.drop(['Sire','Dam' ,'plc'],axis=1)


logging.info('Train data: %s \n %s', np.shape(X_train), X_train.head())
# exit()

X_test = pd.merge(X_test, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])
X_test = pd.merge(X_test, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])
X_test = X_test.drop(['Sire','Dam' ,'plc'],axis=1)


logging.info('Test data:  %s \n %s', np.shape(X_test), X_train.head())

# ------------ select columns for prediction
# Train_Columns = [ 'draw','Age','awt','dhw','J_Win','T_Win']



# X_train = X_train[Train_Columns]
# X_test = X_test[Train_Columns]
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# --------- Fill NaN
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

# --------- Scaler data
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# logging.info('Scaled data %s \n %s', np.shape(X_train), X_train)

# ---------- Regression model
model = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.fit(X_train, y_train.values.ravel())

# ---------- Test Model

X_test = scaler.transform(X_test)  
y_pred = model.predict(X_test)

y_test.loc[:, 'pred_finishTime'] = y_pred

# X_test = scaler.inverse_transform(X_test) 
df_out = pd.merge(X_copy, y_test, how='right',
                      left_index=True, right_index=True)
# logging.info('Original df_out  %s \n %s', np.shape(df_out), df_out.head())

df_out["pred_plc"] = df_out.groupby(['date', 'raceNo'])["pred_finishTime"].rank()
df_out["real_plc"] = df_out.groupby(['date', 'raceNo'])["plc"].rank()

df_out = df_out[['date', 'raceNo','plc','finishTime', 'pred_finishTime','real_plc', 'pred_plc']]

df_out = df_out[(df_out['pred_plc'] <= 1)]

logging.info('df_out: %s \n %s', np.shape(df_out), df_out)

# ----------- Result
print('mean_squared_error: %s ', mean_squared_error(df_out['real_plc'], df_out['pred_plc']))
logging.info('mean_squared_error: %s ', mean_squared_error(df_out['real_plc'], df_out['pred_plc']))




# ------------ Temp ----------
jockey = pd.read_csv('Raw Data/jockey1819.csv', header=0)
trainer = pd.read_csv('Raw Data/trainer1819.csv', header=0)




# ------------ Prediction
pred_data = pd.read_csv('Processed Data/match_data_20190518.csv', header=0)
pred_data = pred_data[(pred_data['dist'] == '1200M') & (pred_data['raceCourse'] == 'ST') & (pred_data['road'].str.contains('TURF')) & (pred_data['going'].str.contains('GOOD'))]
# pred_data_original = pred_data.copy()
pred_data = pred_data.rename(
    columns={
            'AWT': 'awt',
            'Draw': 'draw',
            'Horse Wt. (Declaration)': 'dhw',
            })

pred_data_original = pred_data.copy()
# logging.info('pred_data_original: %s \n %s', np.shape(pred_data_original), pred_data_original)


pred_data = pred_data[['draw', 'Age', 'awt',
                       'dhw', 'Jockey', 'Trainer', 'Sire', 'Dam']]

# logging.info('pred_data: %s \n %s', np.shape(
#     pred_data), pred_data)

pred_data = pd.merge(pred_data, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])
# logging.info('pred_data 1: %s \n %s', np.shape(pred_data), pred_data)

pred_data = pd.merge(pred_data, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])
# logging.info('pred_data 2: %s \n %s', np.shape(pred_data), pred_data)

pred_data = pd.merge(pred_data, jockey[['J_Win','Jockey']], how='left',
                     left_on=['Jockey'], right_on=['Jockey'])

pred_data = pd.merge(pred_data, trainer[['T_Win','Trainer']], how='left',
                     left_on=['Trainer'], right_on=['Trainer'])

pred_data.fillna(X_train.mean(), inplace=True)
pred_data = pred_data.drop(['Sire','Dam','Trainer','Jockey'],axis=1)

# logging.info('pred_data: 3%s \n %s', np.shape(pred_data), pred_data)
pred_data = pred_data.astype(float)

pred_data = scaler.transform(pred_data) 
pred_result = model.predict(pred_data)

pred_data_original['pred_finishTime'] = pred_result

# logging.info('pred_data_original: %s \n %s', np.shape(
#     pred_data_original), pred_data_original)

pred_data_original.loc[:, 'pred_plc'] = pred_data_original.groupby(['raceNo'])[
        "pred_finishTime"].rank()
pred_data_original = pred_data_original[[
        'raceNo', 'Horse No.', 'Horse', 'draw', 'pred_finishTime', 'pred_plc']]
# logging.info('Prediction result: %s \n %s',
#              np.shape(pred_data_original), pred_data_original)


pred_data_original = pred_data_original[(pred_data_original['pred_plc'] <= 1)]

print(pred_data_original)
