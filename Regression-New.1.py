import numpy as np
import pandas as pd
import logging
from multiprocessing import Process, Value, Lock, Pool
import itertools 
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score

# ---------- Parameter
date = '20190529'
# raceCourse = 'ST'
# test_size = 0.01 # For real prediction
# test_size = 0.07 # For backtest predictionx
# test_size = 0.70 # For training
# test_size = 0.60 # For training
# test_size = 0.50 # For training
# test_size = 0.42 # For training
# test_size = 0.40 # For training
# test_size = 0.30 # For training
# test_size = 0.20 # For training
# test_size = 0.10 # For training
dist = '1200M'
# dist = '1650M'
split_date = 20180831

# --------- Setting
logging.basicConfig(filename='./Log/Regression-New.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


"""  
Read required csv files
Convert columns to required format
Combine required csv files into 'data'
"""

data = pd.read_csv('Raw Data/history_bak.csv', header=0,low_memory=False)
raceCard = pd.read_csv('Raw Data/match_data_race_card_bak.csv', header=0,low_memory=False)
# horse = pd.read_csv('Raw Data/horse_bak.csv', header=0)
# jockey = pd.read_csv('Raw Data/jockey1819.csv', header=0)
# trainer = pd.read_csv('Raw Data/trainer1819.csv', header=0)

data = data.iloc[::-1]
data = data[data.finishTime != '---']
data = data[data.plc != 'DISQ']
data = data[data['class'].str.contains('Class') & ~data['class'].str.contains('\(Restricted\)')]
data['plc'] = data['plc'].str.replace('.DH', '', regex=True)
data['finishTime'] = (pd.to_datetime(
    data['finishTime'], format="%M:%S.%f") - datetime(1900, 1, 1))/timedelta(milliseconds=1)

data['plc'] = data['plc'].astype(float)
data['date'] = data['date'].str.replace('/','').astype(float)
data['raceNo'] = data['raceNo'].astype(float)
data['horseNo'] = data['horseNo'].astype(float)
data['odds'] = data['odds'].astype(float)

raceCard['Horse No.'] = raceCard['Horse No.'].astype(float)
raceCard['date'] = raceCard['date'].astype(float)
raceCard['raceNo'] = raceCard['raceNo'].astype(float)
raceCard['Rtg.+/-'] = raceCard['Rtg.+/-'].replace({'-': np.nan})
raceCard['Rtg.+/-'].fillna(0, inplace=True)
raceCard['Wt.+/- (vs Declaration)'] = raceCard['Wt.+/- (vs Declaration)'].replace({'-': np.nan})
raceCard['Wt.+/- (vs Declaration)'].fillna(0, inplace=True)
raceCard['class'] = raceCard['class'].str.replace('Class ', '', regex=True)

data = pd.merge(data[['finishTime','date','raceNo','horseNo','odds','plc']], raceCard, how='left',
                     left_on=['date','raceNo','horseNo'], right_on=['date','raceNo','Horse No.'])

logging.info('Combine required csv files into "data" %s \n %s', np.shape(data), data.head(1))


""" 
Adding new columns to data for further train, test, predict
Convert category data to numeric
"""

split_result = data["Last 6 Runs"].str.split("/", expand=True)
split_result.columns = ['Runs_1','Runs_2','Runs_3','Runs_4','Runs_5','Runs_6']
split_result = split_result.replace('-',np.nan)
data = pd.merge(data, split_result, how='right',
                      left_index=True, right_index=True)


split_result = data["Gear"].str.split("/", expand=True)
split_result = split_result.replace(to_replace='.-|1|2', value=np.nan,regex=True)
split_result.columns = ['Gear_1','Gear_2','Gear_3','Gear_4']
data = pd.merge(data, split_result, how='right',
                      left_index=True, right_index=True)

col_name= split_result['Gear_1'].value_counts().reset_index()
col_name.columns = ['col_name','count']

for index,row in col_name.iterrows():
    data[row['col_name']] = np.where((data['Gear_1'] ==row['col_name'])|(data['Gear_2'] ==row['col_name'])|(data['Gear_3'] ==row['col_name'])|(data['Gear_4'] ==row['col_name']),1,0)


data = pd.get_dummies(data, columns=[
    'Sex'], prefix=['Sex'])
data = pd.get_dummies(data, columns=[
    'going'], prefix=['going'])
data = pd.get_dummies(data, columns=[
    'raceCourse'], prefix=['raceCourse'])
# data = pd.get_dummies(data, columns=[
#     'dist'], prefix=['dist'])

logging.info('Added columns %s \n %s',
             np.shape(data), data.head(1))


""" 
Select requried data for train, test, predict
"""

data = data[(data['dist'] == dist) & (data['road'].str.contains('TURF'))]
# data = data[data['road'].str.contains('TURF',na=False)]
q = data["finishTime"].quantile(0.99)
data = data[data["finishTime"] < q]
# data_original = data.copy() 

logging.info('Selected data %s \n %s', np.shape(data), data.head(2).append(data.tail(2)))


""" 
Split data for train, test, predict
Add Column Jockey Rank, Trainer Rank
Add Column Dam Rank, Sire Rank
Add Column Horse Rank
"""

X = data
X_copy = X.copy() 
y = data[['finishTime']]

# X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, shuffle=False)



X_train, X_test = data[data['date'] < split_date].drop(
    ['finishTime'], axis=1), data[data['date'] >= split_date].drop(['finishTime'], axis=1)
y_train, y_test = data[data['date'] < split_date][['finishTime']], data[data['date']
                                                             >= split_date][['finishTime']]

# ---- Sire Rank
sireRank = X_train.groupby(['Sire'])['plc'].apply(lambda x: (x<=3).sum())
sireRank = sireRank.reset_index()
sireRank.columns = ['Sire','SireRank']
X_train = pd.merge(X_train, sireRank, how='left',
    left_on=['Sire'], right_on=['Sire'])

# ---- Dam Rank
damRank = X_train.groupby(['Dam'])['plc'].apply(lambda x: (x<=3).sum())
damRank = damRank.reset_index()
damRank.columns = ['Dam','DamRank']
X_train = pd.merge(X_train, damRank, how='left',
    left_on=['Dam'], right_on=['Dam'])


# ---- Horse Rank
horseRank = X_train.groupby(['Brand No.'])['plc'].apply(lambda x: (x<=3).sum())/ X_train.groupby(['Brand No.'])['plc'].count()
horseRank = horseRank.reset_index()
horseRank.columns = ['Brand No.','horseRank']
X_train = pd.merge(X_train, horseRank[['horseRank','Brand No.']], how='left',
    left_on=['Brand No.'], right_on=['Brand No.'])
X_train['HorseMatchRank'] = X_train.groupby(['date', 'raceNo'])[
    "horseRank"].rank()

# ---- Jockey Rank
jockeyRank = X_train.groupby(['Jockey'])['plc'].apply(lambda x: (x<=3).sum())/ X_train.groupby(['Jockey'])['plc'].count()
jockeyRank = jockeyRank.reset_index()
jockeyRank.columns = ['Jockey','JockeyRank']
X_train = pd.merge(X_train, jockeyRank[['JockeyRank','Jockey']], how='left',
    left_on=['Jockey'], right_on=['Jockey'])


# ---- Trainer Rank
trainerRank = X_train.groupby(['Trainer'])['plc'].apply(lambda x: (x<=3).sum()) / X_train.groupby(['Trainer'])['plc'].count()
trainerRank = trainerRank.reset_index()
trainerRank.columns = ['Trainer','TrainerRank']
X_train = pd.merge(X_train, trainerRank[['TrainerRank','Trainer']], how='left',
    left_on=['Trainer'], right_on=['Trainer'])


logging.info('X_train data %s \n %s', np.shape(X_train), X_train.head(2).append(X_train.tail(2)))


""" 
Select requried columns for train, test, predict 
"""

# 0.8 /0.375 test split_date=20180831  data=20190529 odds=2-6 andom_state=1, solver='lbfgs' 1200M
# train_test_col = ['Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'Runs_5', 'Runs_6', 'B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'raceCourse_ST', 'Draw', 'Rtg.+/-', 'AWT', 'Horse Wt. (Declaration)', 'class', 'DamRank', 'Age']

#Origial
train_test_col = ['B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'raceCourse_ST', 'Runs_6', 'Runs_5', 'Runs_4', 'Runs_3', 'Runs_2', 'Runs_1','TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'Draw', 'Rtg.+/-', 'AWT', 'class', 'DamRank', 'HorseMatchRank', 'Age', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)']


# train_test_col = ['H', 'Sex_f', 'Rtg.+/-', 'Wt.+/- (vs Declaration)']

# train_test_col = ['Runs_1', 'PS', 'Rtg.+/-']
# train_test_col =  ['Wt.+/- (vs Declaration)', 'Age', 'class', 'Rtg.+/-', 'SB',]
# 0.7894 /0.4211 test split_date=20180831  data=20190526 odds=2-6 andom_state=1, solver='lbfgs' 1650M
# train_test_col = ['B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'Runs_5', 'Runs_6', 'raceCourse_ST', 'Draw', 'Age', 'AWT', 'Rtg.+/-', 'DamRank', 'Horse Wt. (Declaration)', 'class']

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()
y_train_copy = y_train.copy()




train_test_col = list(train_test_col)
X_train = X_train_copy[train_test_col]

X_train = X_train.astype(float)
# print(train_test_col)
X_train = X_train[train_test_col]

X_train = X_train.astype(float)


# ---- save columns for further test or prediciton 
predictionColumns = X_train.columns.values


# --------- Fill all missing data
print(X_train.head())
print(X_train.mean())
X_train.fillna(X_train.mean(), inplace=True)
print(X_train.head())
X_train_backup = X_train

headers = ','.join(map(str, X_train.columns.values))
np.savetxt('./Processed Data/trainX_'+date+'.csv', X_train,
                   delimiter=',', fmt='%s', header=headers, comments='')
# X_test.fillna(X_train.mean(), inplace=True)
# logging.info('Test data filled NaN:  %s \n %s' , np.shape(X_test), X_test.head(2).append( X_test.tail(2)))
    
# print(X_train.head())
""" 
Scale data
Train model
"""
# --------- Scaler data
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)


# ---------- Regression model
# model = MLPRegressor(hidden_layer_sizes=(5,5,5,5,5,5),activation='relu', solver='adam', alpha=0.00001, shuffle=True, random_state=13,learning_rate='constant',max_iter=3000) # 0.74 Test size 50%
# model = MLPRegressor(random_state=13)

model = MLPRegressor(random_state=1, solver='lbfgs') 
model.fit(X_train, y_train.values.ravel())


""" 
Test the model with X_test
Generate Test result report
"""

# ---- Set up for X_test
X_test = pd.merge(X_test_copy, horseRank[['horseRank', 'Brand No.']], how='left',
                left_on=['Brand No.'], right_on=['Brand No.'])
X_test = pd.merge(X_test, sireRank, how='left',
                left_on=['Sire'], right_on=['Sire'])
X_test = pd.merge(X_test, damRank, how='left',
                left_on=['Dam'], right_on=['Dam'])
X_test = pd.merge(X_test, jockeyRank[['JockeyRank', 'Jockey']], how='left',
                left_on=['Jockey'], right_on=['Jockey'])
X_test = pd.merge(X_test, trainerRank[['TrainerRank', 'Trainer']], how='left',
                left_on=['Trainer'], right_on=['Trainer'])
X_test['HorseMatchRank'] = X_test.groupby(['date', 'raceNo'])[
    "horseRank"].rank()

# ---- Fill missing data
X_test.fillna(X_train_backup.mean(), inplace=True)

# print(X_test)
headers = ','.join(map(str, X_test.columns.values))
np.savetxt('./Processed Data/testX_'+date+'.csv', X_test,
                   delimiter=',', fmt='%s', header=headers, comments='')

headers = ','.join(map(str, y_test.columns.values))
np.savetxt('./Processed Data/testY_'+date+'.csv', y_test,
                   delimiter=',', fmt='%s', header=headers, comments='')

headers = ','.join(map(str, y_train.columns.values))
np.savetxt('./Processed Data/trainY_'+date+'.csv', y_train,
                   delimiter=',', fmt='%s', header=headers, comments='')
# ---- Select required columns
X_test = X_test[predictionColumns]

# ---- scale data and test will trianed model
X_test = X_test.astype(float)
X_test = scaler.transform(X_test)  
y_pred = model.predict(X_test)

# ---- Prepare data for report generation
y_test.loc[:, 'pred_finishTime'] = y_pred
test_report = pd.merge(X_copy, y_test, how='right',
                    left_index=True, right_index=True)

test_report["pred_plc"] = test_report.groupby(['date', 'raceNo'])["pred_finishTime"].rank()
test_report["real_plc"] = test_report.groupby(['date', 'raceNo'])["plc"].rank()

# ---- Generate Report
test_report = test_report[['date', 'raceNo','horseNo', 'plc','odds', 'pred_finishTime','real_plc', 'pred_plc',]]

headers = ','.join(map(str, test_report.columns.values))
np.savetxt('./Report/test_result_'+date+'.csv', test_report.round(0),
                delimiter=',', fmt='%s', header=headers, comments='')


# ---- Accuracy rate
# test_report = test_report[(test_report['pred_plc'] <= 1) & (test_report['odds'].astype(float) <= 6) & (test_report['odds'].astype(float) >= 2)]
test_report = test_report[(test_report['pred_plc'] <= 1) & (test_report['odds'].astype(float) <= 15) & (test_report['odds'].astype(float) >= 5)]
# test_report = test_report[(test_report['pred_plc'] <= 1) ]

test_report.loc[test_report['real_plc'] <=3, 'real_first_3'] = 1
test_report.fillna(0,inplace=True)
logging.info('test_report: %s \n %s', np.shape(test_report), test_report)

print('Accuracy score for 1st: %s', accuracy_score(test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
print('Accuracy score for first 3: %s ', accuracy_score(test_report['real_first_3'], test_report['pred_plc']))

print()

logging.info('Accuracy score for 1st: %s', accuracy_score(
    test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
logging.info('Accuracy score for first 3: %s ', accuracy_score(
    test_report['real_first_3'], test_report['pred_plc']))

test_report = test_report.tail(20)
print('Accuracy score for 1st recent 20 matches: %s', accuracy_score(
    test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
print('Accuracy score for first 3 recent 20 matches: %s ', accuracy_score(
    test_report['real_first_3'], test_report['pred_plc']))
print()

logging.info('Accuracy score for 1st recent 20 matches: %s', accuracy_score(
    test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
logging.info('Accuracy score for first 3 recent 20 matches: %s ',
            accuracy_score(test_report['real_first_3'], test_report['pred_plc']))


test_report = test_report.tail(10)
print('Accuracy score for 1st recent 10 matches: %s', accuracy_score(test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
print('Accuracy score for first 3 recent 10 matches: %s ', accuracy_score(test_report['real_first_3'], test_report['pred_plc']))
print()

logging.info('Accuracy score for 1st recent 10 matches: %s', accuracy_score(
    test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
logging.info('Accuracy score for first 3 recent 10 matches: %s ',
            accuracy_score(test_report['real_first_3'], test_report['pred_plc']))



# exit()












""" 
Prediction
Predict coming race with the model 
Generate prediction report
"""
# ---- Read csv
match_data_race_card = pd.read_csv(
    'Raw Data/match_data_race_card.csv', header=0)

# ---- Select required data and convert format
pred_data = match_data_race_card[(match_data_race_card['dist'] == dist) & (match_data_race_card['road'].str.contains('TURF'))]
# pred_data = match_data_race_card[match_data_race_card['road'].str.contains('TURF',na=False)]
pred_data = pred_data[pred_data['class'].str.contains(
    'Class') & ~pred_data['class'].str.contains('\(Restricted\)')]
pred_data['date'] = pred_data['date'].astype(float)
pred_data['Rtg.+/-'] = pred_data['Rtg.+/-'].replace({'-': np.nan})
pred_data['Rtg.+/-'].fillna(0, inplace=True)
pred_data['Wt.+/- (vs Declaration)'] = pred_data['Wt.+/- (vs Declaration)'].replace({'-': np.nan})
pred_data['Wt.+/- (vs Declaration)'].fillna(0, inplace=True)
pred_data['Age'] = pred_data['Age'] + 0.0

# ---- Adding new columns to data for further train, test, predict
# Convert category data to numeric
split_result = pred_data["Last 6 Runs"].str.split("/", expand=True)
split_result.columns = ['Runs_1', 'Runs_2',
                        'Runs_3', 'Runs_4', 'Runs_5', 'Runs_6']
split_result = split_result.replace('-', np.nan)
pred_data = pd.merge(pred_data, split_result, how='right',
                left_index=True, right_index=True)


split_result = pred_data["Gear"].str.split("/", expand=True)
split_result = split_result.replace(
    to_replace='.-|1|2', value=np.nan, regex=True)
split_result.columns = ['Gear_1', 'Gear_2', 'Gear_3', 'Gear_4']
pred_data = pd.merge(pred_data, split_result, how='right',
                left_index=True, right_index=True)

for index,row in col_name.iterrows():
    pred_data[row['col_name']] = np.where((pred_data['Gear_1'] == row['col_name']) | (pred_data['Gear_2'] == row['col_name']) | (
        pred_data['Gear_3'] == row['col_name']) | (pred_data['Gear_4'] == row['col_name']), 1, 0)


pred_data['class'] = pred_data['class'].str.replace('Class ','')
pred_data = pd.get_dummies(pred_data, columns=[
    'Sex'], prefix=['Sex'])

pred_data = pd.get_dummies(pred_data, columns=[
    'going'], prefix=['going'])

pred_data = pd.get_dummies(pred_data, columns=[
    'raceCourse'], prefix=['raceCourse'])

# pred_data = pd.get_dummies(pred_data, columns=[
    # 'dist'], prefix=['dist'])

pred_data = pd.merge(pred_data, horseRank[['horseRank', 'Brand No.']], how='left',
                     left_on=['Brand No.'], right_on=['Brand No.'])
pred_data = pd.merge(pred_data, sireRank, how='left',
                     left_on=['Sire'], right_on=['Sire'])
pred_data = pd.merge(pred_data, damRank, how='left',
                     left_on=['Dam'], right_on=['Dam'])
pred_data = pd.merge(pred_data, jockeyRank[['JockeyRank', 'Jockey']], how='left',
                     left_on=['Jockey'], right_on=['Jockey'])
pred_data = pd.merge(pred_data, trainerRank[['TrainerRank', 'Trainer']], how='left',
                     left_on=['Trainer'], right_on=['Trainer'])

# ---- Fill missing data
pred_data.fillna(X_train_backup.mean(), inplace=True)

# ---- Generate missing columns
for r in predictionColumns:
    if r not in pred_data:
        pred_data[r] = np.NaN

# ---- Fill missing columns values
pred_data.fillna(0, inplace=True)

pred_data_original = pred_data.copy()
logging.info('Prediction record : %s \n %s', np.shape(pred_data),
             pred_data.head(2).append(pred_data.tail(2)))
# ---- Selected required only
pred_data = pred_data[predictionColumns.tolist()]

logging.info('Prediction data : %s \n %s', np.shape(pred_data), pred_data.head(2).append(pred_data.tail(2)))


# ---- Prediction 
pred_data = pred_data.astype(float)
pred_data = scaler.transform(pred_data) 
pred_result = model.predict(pred_data)


# ---- Report generate
pred_data_original['pred_finishTime'] = pred_result
pred_data_original["pred_plc"] = pred_data_original.groupby(['date', 'raceNo'])[
    "pred_finishTime"].rank()

logging.info('Prediction Result: %s \n %s', np.shape(
    pred_data_original), pred_data_original.head(2).append(pred_data_original.tail(2)))




prediction_report = pred_data_original[[
        'date','raceNo', 'Horse No.', 'Horse', 'pred_finishTime', 'pred_plc']]
# logging.info('Prediction result: %s \n %s',
#              np.shape(pred_data_original), pred_data_original)

headers = ','.join(map(str, prediction_report.columns.values))
np.savetxt('./Report/regression_result_'+date+'.csv', prediction_report.round(0),
                   delimiter=',', fmt='%s', header=headers, comments='')

prediction_report = prediction_report[(prediction_report['pred_plc'] <= 1)]
logging.info('Prediction result: %s \n %s',
             np.shape(prediction_report), prediction_report)
print(prediction_report)
