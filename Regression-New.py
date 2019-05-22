import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score

# ---------- Parameter
date = '20190522'
raceCourse = 'HV'
test_size = 0.20
dist = '1200M'


# --------- Setting
logging.basicConfig(filename='./Log/Regression-New.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# --------- Read CSV
data = pd.read_csv('Raw Data/history_bak.csv', header=0)
horse = pd.read_csv('Raw Data/horse_bak.csv', header=0)
jockey = pd.read_csv('Raw Data/jockey1819.csv', header=0)
trainer = pd.read_csv('Raw Data/trainer1819.csv', header=0)

data = data.iloc[::-1]
data = data[data.finishTime != '---']
data = data[data.plc != 'DISQ']
data = data[data['class'].str.contains('Class') & ~data['class'].str.contains('\(Restricted\)')]
data['plc'] = data['plc'].str.replace('.DH', '', regex=True)
data['finishTime'] = (pd.to_datetime(
    data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)
data['plc'] = data['plc'].astype(float)

logging.info('Original data %s \n %s', np.shape(data), data.head())

data = pd.merge(data, trainer, how='left',
                     left_on=['trainer'], right_on=['Trainer'])
data = pd.merge(data, jockey, how='left',
                     left_on=['jockey'], right_on=['Jockey'])


split_result = data["horseName"].str.split("(", n=1, expand=True)
data["horseName"] = split_result[0]
data["horseCode"] = split_result[1].str.replace(')', '')

data = pd.merge(data, horse[['Sire','Dam','Code','Foaling Date']], how='left',
                     left_on=['horseCode'], right_on=['Code'])

data['Age'] = (pd.to_datetime(data['date'],format='%Y/%m/%d', errors='ignore') - pd.to_datetime(data['Foaling Date'],format='%d/%m/%Y', errors='ignore'))/np.timedelta64(1,'Y')

data.dropna(subset=['Age'],inplace=True)

logging.info('After Merged data head : %s \n %s', np.shape(data), data.head())
logging.info('After Merged data tail : %s \n %s', np.shape(data), data.tail())



# --------- Select data for prediction
data = data[(data['dist'] == dist) & (data['raceCourse'] == raceCourse) & (data['road'].str.contains('TURF')) ]
q = data["finishTime"].quantile(0.99)
data = data[data["finishTime"] < q]
data_original = data.copy() 

logging.info('Selected data %s \n %s', np.shape(data), data.head())

# exit()

# --------- Prepare Train Test
X = data.drop(['finishTime'], axis=1)
X_copy = X.copy() 
y = data[['finishTime']]


X = X[[ 'draw','Age','awt','dhw','J_Win','T_Win','Sire','Dam','plc','T_2nd','J_2nd','J_3rd','J_4th','Total Rides','T_3rd','T_4th','Total Runs','class','going']]

X['class'] = X['class'].str.replace('Class ','')

# X = pd.get_dummies(X, columns=[
#     'Age'], prefix=['Age'])

X = pd.get_dummies(X, columns=[
    'going'], prefix=['going'])

X = pd.get_dummies(X, columns=[
    'draw'], prefix=['draw'])

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)

X_train['finishTime'] = y_train
sumOfPlc = X_train.groupby(['Sire'])['plc'].sum()
noOfMatch = X_train.groupby(['Sire'])['finishTime'].count()
sireRank = sumOfPlc/noOfMatch
sireRank = sireRank.reset_index()
sireRank.columns = ['Sire','SireRank']

X_train = pd.merge(X_train, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])

sumOfPlc = X_train.groupby(['Dam'])['plc'].sum()
noOfMatch = X_train.groupby(['Dam'])['finishTime'].count()
damRank = sumOfPlc/noOfMatch
damRank = damRank.reset_index()
damRank.columns = ['Dam','DamRank']

X_train = pd.merge(X_train, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])

X_train['JockeyRank'] = (X_train['J_Win']*5 + X_train['J_2nd']*4 + X_train['J_3rd']*3 + X_train['J_4th']*2)
X_train['TrainerRank'] = (X_train['T_Win']*5 + X_train['T_2nd']*4 + X_train['T_3rd']*3 + X_train['T_4th']*2) 

X_train = X_train.drop(['finishTime','Sire','Dam' ,'plc','J_Win','J_2nd','J_3rd','Total Rides','T_Win','T_2nd','T_3rd','Total Runs'],axis=1)


# logging.info('Train data: %s \n %s', np.shape(X_train), X_train.head())
# exit()




X_test = pd.merge(X_test, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])
X_test = pd.merge(X_test, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])

X_test['JockeyRank'] = (X_test['J_Win']*5 + X_test['J_2nd']*4 + X_test['J_3rd']*3 + X_test['J_4th']*2) 
X_test['TrainerRank'] = (X_test['T_Win']*5 + X_test['T_2nd']*4 + X_test['T_3rd']*3 + X_test['T_4th']*2)

X_test = X_test.drop(['Sire','Dam' ,'plc','J_Win','J_2nd','J_3rd','Total Rides','T_Win','T_2nd','T_3rd','Total Runs'],axis=1)

logging.info('Train data before filled NaN:  %s \n %s', np.shape(X_train), X_train.head())
logging.info('Test data before filled NaN:  %s \n %s', np.shape(X_test), X_test.head())


# ------------ select columns for prediction
# Train_Columns = [ 'draw','Age','awt','dhw','J_Win','T_Win']



# X_train = X_train[Train_Columns]
# X_test = X_test[Train_Columns]
X_train = X_train.astype(float)
X_test = X_test.astype(float)
predictionColumns = X_train.columns.values

# --------- Fill NaN
X_train_backup = X_train
# print(X_train['JockeyRank'].describe())
# print(X_train['TrainerRank'].describe())
# print(X_train['TrainerRank'].quantile(0.25))

X_train['JockeyRank'].fillna(X_train['JockeyRank'].quantile(0.25), inplace=True)
X_test['JockeyRank'].fillna(X_train['JockeyRank'].quantile(0.25), inplace=True)
X_train['TrainerRank'].fillna(X_train['TrainerRank'].quantile(0.25), inplace=True)
X_test['TrainerRank'].fillna(X_train['TrainerRank'].quantile(0.25), inplace=True)
# X_test['DamRank'].fillna(X_train['DamRank'].quantile(0.75), inplace=True)
# X_test['SireRank'].fillna(X_train['SireRank'].quantile(0.75), inplace=True)

# print(X_test.isnull().values.any())
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)
logging.info('Train data filled NaN:  %s \n %s', np.shape(X_train), X_train.head())
logging.info('Test data filled NaN:  %s \n %s', np.shape(X_test), X_test.head())
    

# --------- Scaler data
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# logging.info('Scaled data %s \n %s', np.shape(X_train), X_train)

# ---------- Regression model
model = MLPRegressor(hidden_layer_sizes=(5,5,5,),activation='relu', solver='lbfgs', alpha=0.0001, shuffle=True, random_state=10,learning_rate='constant'
    )

model.fit(X_train, y_train.values.ravel())

# model = MLPRegressor()
# params = {'solver':['lbfgs'],'shuffle':[True],'activation':['relu'],'alpha':[0.001,0.0001,0.00001],'hidden_layer_sizes':[(10,),(20,),(30,)]}
# gs = GridSearchCV(model, params,cv=3,n_jobs=-1)

# gs.fit(X_train, y_train.values.ravel())
# # X_test = scaler.transform(X_test)
# model= gs
# print(gs.best_score_)#最好的得分

# print(gs.best_params_)#最好的参数
# print(pred)
# exit()

# print(model.score)





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

df_out = df_out[['date', 'raceNo','horseNo', 'plc','finishTime', 'pred_finishTime','real_plc', 'pred_plc']]
df_out_original = df_out.copy()
df_out = df_out[(df_out['pred_plc'] <= 1)]
df_out.loc[df_out['real_plc'] <=3, 'real_first_3'] = 1
df_out.fillna(0,inplace=True)
# q = df_out["plc"].quantile(0.99)
# df_out = df_out[df_out["plc"] < q]
logging.info('df_out: %s \n %s', np.shape(df_out), df_out)

# ----------- Result
# print('mean_squared_error for finishTime: %s ', mean_squared_error(df_out_original['finishTime'], df_out_original['pred_finishTime']))
# logging.info('mean_squared_error for finishTime: %s ', mean_squared_error(df_out_original['real_plc'], df_out_original['pred_plc']))

# print('mean_squared_error for 1st: %s ', mean_squared_error(df_out['real_plc'], df_out['pred_plc']))
# logging.info('mean_squared_error for 1st: %s ', mean_squared_error(df_out['real_plc'], df_out['pred_plc']))

# print('mean_squared_error for first 3: %s ', mean_squared_error(df_out['real_first_3'], df_out['pred_plc']))
# logging.info('mean_squared_error for first 3: %s ', mean_squared_error(df_out['real_first_3'], df_out['pred_plc']))

print('Accuracy score for 1st: %s', accuracy_score(df_out['real_plc'], df_out['pred_plc']))
print('Accuracy score for first 3: %s ', accuracy_score(df_out['real_first_3'], df_out['pred_plc']))
print()

df_out = df_out.head(10)
print('Accuracy score for 1st coming 10 matches: %s', accuracy_score(df_out['real_plc'], df_out['pred_plc']))
print('Accuracy score for first 3 coming 10 matches: %s ', accuracy_score(df_out['real_first_3'], df_out['pred_plc']))
print()

headers = ','.join(map(str, df_out_original.columns.values))
np.savetxt('./Report/test_result_'+date+'.csv', df_out_original.round(0),
                   delimiter=',', fmt='%s', header=headers, comments='')







# ------------ Prediction
pred_data = pd.read_csv('Processed Data/match_data_'+date+'.csv', header=0)
pred_data = pred_data[(pred_data['dist'] == dist) & (pred_data['raceCourse'] == raceCourse) & (pred_data['road'].str.contains('TURF') )]
pred_data = pred_data.iloc[::-1]
pred_data = pred_data.rename(
    columns={
            'AWT': 'awt',
            'Draw': 'draw',
            'Horse Wt. (Declaration)': 'dhw',
            })

pred_data_original = pred_data.copy()
# logging.info('pred_data_original: %s \n %s', np.shape(pred_data_original), pred_data_original)


pred_data = pred_data[['draw', 'awt',
                       'dhw', 'Jockey', 'Trainer', 'Sire', 'Dam','Brand No.','date','Age','going','class']]

pred_data['Age'] = pred_data['Age'] +0.0

pred_data['class'] = pred_data['class'].str.replace('Class ','')
pred_data = pd.get_dummies(pred_data, columns=[
    'draw'], prefix=['draw'])

# pred_data = pd.get_dummies(pred_data, columns=[
#     'Age'], prefix=['Age'])

pred_data = pd.get_dummies(pred_data, columns=[
    'going'], prefix=['going'])

# logging.info('pred_data: %s \n %s', np.shape(
#     pred_data), pred_data)

pred_data = pd.merge(pred_data, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])
# logging.info('pred_data 1: %s \n %s', np.shape(pred_data), pred_data)

pred_data = pd.merge(pred_data, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])
# logging.info('pred_data 2: %s \n %s', np.shape(pred_data), pred_data)

pred_data = pd.merge(pred_data, jockey, how='left',
                     left_on=['Jockey'], right_on=['Jockey'])

pred_data = pd.merge(pred_data, trainer, how='left',
                     left_on=['Trainer'], right_on=['Trainer'])

pred_data = pd.merge(pred_data, horse[['Code','Foaling Date']], how='left',
                     left_on=['Brand No.'], right_on=['Code'])

pred_data['Age'] = (pd.to_datetime(pred_data['date'],format='%Y%m%d', errors='ignore') - pd.to_datetime(pred_data['Foaling Date'],format='%d/%m/%Y', errors='ignore'))/np.timedelta64(1,'Y')
pred_data['JockeyRank'] = (pred_data['J_Win']*5 + pred_data['J_2nd']*4 + pred_data['J_3rd']*3 + pred_data['J_4th']*2)
pred_data['TrainerRank'] = (pred_data['T_Win']*5 + pred_data['T_2nd']*4 + pred_data['T_3rd']*3 + pred_data['T_4th']*2)

pred_data = pred_data.drop(['Sire','Dam','Trainer','Jockey','Brand No.','date','Code','Foaling Date'],axis=1)
pred_data['JockeyRank'].fillna(X_train_backup['JockeyRank'].quantile(0.25), inplace=True)
pred_data['TrainerRank'].fillna(X_train_backup['TrainerRank'].quantile(0.25), inplace=True)

pred_data.fillna(X_train_backup.mean(), inplace=True)

for r in predictionColumns:
    if r not in pred_data:
        pred_data[r] = np.NaN
pred_data = pred_data[predictionColumns]

pred_data.fillna(0, inplace=True)


logging.info('Prediction data filled NaN: %s \n %s', np.shape(pred_data), pred_data.head())
pred_data = pred_data.astype(float)

pred_data = scaler.transform(pred_data) 
pred_result = model.predict(pred_data)

pred_data_original['pred_finishTime'] = pred_result

# logging.info('pred_data_original: %s \n %s', np.shape(
#     pred_data_original), pred_data_original)

pred_data_original.loc[:, 'pred_plc'] = pred_data_original.groupby(['raceNo'])[
        "pred_finishTime"].rank()
pred_data_original = pred_data_original[[
        'date','raceNo', 'Horse No.', 'Horse', 'pred_finishTime', 'pred_plc']]
# logging.info('Prediction result: %s \n %s',
#              np.shape(pred_data_original), pred_data_original)

headers = ','.join(map(str, pred_data_original.columns.values))
np.savetxt('./Report/regression_result_'+date+'.csv', pred_data_original.round(0),
                   delimiter=',', fmt='%s', header=headers, comments='')

pred_data_original = pred_data_original[(pred_data_original['pred_plc'] <= 1)]
logging.info('Prediction result: %s \n %s',
             np.shape(pred_data_original), pred_data_original)
print(pred_data_original)
