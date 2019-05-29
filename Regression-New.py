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
# dist = '1200M'
dist = '1650M'
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
raceCard['class'] = raceCard['class'].str.replace('Class ', '', regex=True)

data = pd.merge(data[['finishTime','date','raceNo','horseNo','odds','plc']], raceCard, how='left',
                     left_on=['date','raceNo','horseNo'], right_on=['date','raceNo','Horse No.'])

# logging.info('Combine required csv files into "data" %s \n %s', np.shape(data), data.head(1))


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
# logging.info('Added columns %s \n %s',
#              np.shape(data), data.head(1))


""" 
Select requried data for train, test, predict
"""

data = data[(data['dist'] == dist) & (data['road'].str.contains('TURF'))]
# data = data[data['road'].str.contains('TURF',na=False)]
q = data["finishTime"].quantile(0.99)
data = data[data["finishTime"] < q]
# data_original = data.copy() 

# logging.info('Selected data %s \n %s', np.shape(data), data.head(2).append(data.tail(2)))


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


""" 
Select requried columns for train, test, predict 
"""


train_test_col = ['B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'TrainerRank', 'SireRank', 'horseRank', 'JockeyRank','Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'Runs_5', 'Runs_6', 'raceCourse_ST', 'Draw', 'Rtg.+/-', 'Age', 'AWT', 'class', 'DamRank', 'Horse Wt. (Declaration)']



score_first_3 = 0
score_first_1 = 0
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()
y_train_copy = y_train.copy()

def init(args):
    ''' store the counter for later use '''
    global score_first_1
    global score_first_3
    global X_train_copy
    global X_test_copy
    global y_train_copy

def train(train_test_col):
    global score_first_1
    global score_first_3
    global X_train_copy
    global X_test_copy
    global y_train_copy

    train_test_col = list(train_test_col)
    X_train = X_train_copy[train_test_col]

    X_train = X_train.astype(float)

    X_train = X_train[train_test_col]

    X_train = X_train.astype(float)


    # ---- save columns for further test or prediciton 
    predictionColumns = X_train.columns.values


    # --------- Fill all missing data
    X_train_backup = X_train
    X_train.fillna(X_train.mean(), inplace=True)
        

    """ 
    Scale data
    Train model
    """
    # --------- Scaler data
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)


    # ---------- Regression model
    # model = MLPRegressor(solver='lbfgs') # 0.64 Test size 70%
    model = MLPRegressor(random_state=1, solver='lbfgs') 
    # model = MLPRegressor(hidden_layer_sizes=(3,3,3,3),activation='relu', solver='lbfgs', alpha=0.0001, shuffle=True,) 
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

    # ---- Fill missing data
    X_test.fillna(X_train_backup.mean(), inplace=True)

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

    # # ---- Generate Report
    test_report = test_report[['date', 'raceNo','horseNo', 'plc','odds', 'pred_finishTime','real_plc', 'pred_plc',]]

    # headers = ','.join(map(str, test_report.columns.values))
    # np.savetxt('./Report/test_result_'+date+'.csv', test_report.round(0),
    #                 delimiter=',', fmt='%s', header=headers, comments='')


    # ---- Accuracy rate
    test_report = test_report[(test_report['pred_plc'] <= 1) & (test_report['odds'].astype(float) <= 6) & (test_report['odds'].astype(float) >= 2)]
    # test_report = test_report[(test_report['pred_plc'] <= 1) ]

    test_report.loc[test_report['real_plc'] <=3, 'real_first_3'] = 1
    test_report.fillna(0,inplace=True)

    with score_first_3.get_lock(),score_first_1.get_lock():
        
        avg_original = (score_first_3.value + score_first_1.value)/2
        avg_current = (accuracy_score(test_report['real_first_3'], test_report['pred_plc']) + accuracy_score(test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))/2
        # print(avg_original)
        # print(avg_current)
        # if (accuracy_score(test_report['real_first_3'], test_report['pred_plc']) > score_first_3.value) | (accuracy_score(test_report['real_plc'].round(0), test_report['pred_plc'].round(0)) > score_first_1.value):
        if avg_current > avg_original :
            score_first_3_original = score_first_3.value
            score_first_1_original = score_first_1.value
            score_first_3.value = accuracy_score(test_report['real_first_3'], test_report['pred_plc'])
            score_first_1.value = accuracy_score(test_report['real_plc'].round(0), test_report['pred_plc'].round(0))
            # col = train_test_col

            print('score_first_3 updated from ',score_first_3_original,' to ', score_first_3.value)
            print('score_first_1 updated from ',score_first_1_original,' to ', score_first_1.value)
            print('Columns: \n %s', train_test_col)
            logging.info('score and col updated score_first_1: %s,  score_first_3: %s \n %s',score_first_1.value,score_first_3.value,train_test_col )



perm = itertools.permutations(train_test_col)
if __name__ == "__main__":
    names = perm
    procs = []
    score_first_1 = Value('f', 0)
    score_first_3 = Value('f', 0)

    pool = Pool(processes=3)
    for name in names:
        pool.map_async(train, (name,))
    pool.close()
    pool.join()

exit()