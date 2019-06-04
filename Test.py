import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import logging
import itertools
from sklearn.metrics import mean_squared_error, accuracy_score
from multiprocessing import Process, Value, Lock, Pool, Manager
import time

# Config
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler("{0}/{1}.log".format('./Log/', 'Test'))])

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


date = '20190602'

testX = pd.read_csv('Processed Data/testX_'+date+'.csv',
                    header=0, low_memory=False)
testY = pd.read_csv('Processed Data/testY_'+date+'.csv',
                    header=0, low_memory=False)
trainX = pd.read_csv('Processed Data/trainX_'+date+'.csv',
                     header=0, low_memory=False)
trainY = pd.read_csv('Processed Data/trainY_'+date+'.csv',
                     header=0, low_memory=False)
pred = pd.read_csv('Processed Data/pred_'+date+'.csv',
                     header=0, low_memory=False)
testX_bak = testX.copy()


def test(train_test_col, odds_max=99, odds_min=1):
    # print('method init')
    global testX
    global testY
    global trainX
    global trainY
    global testX_bak
    train_test_col = list(train_test_col)

    trainX_copy = trainX.copy()
    testX_copy = testX.copy()
    testY_copy = testY.copy()

    trainX_copy = trainX_copy[train_test_col]
    trainX_copy = trainX_copy.astype(float)
    scaler = StandardScaler()
    scaler.fit(trainX_copy)
    trainX_copy = scaler.transform(trainX_copy)
    model = MLPRegressor(random_state=1, solver='lbfgs')
    model.fit(trainX_copy, trainY.values.ravel())

    testX_copy = testX_copy[train_test_col]
    # print(testX_copy)
    testX_copy = testX_copy.astype(float)
    testX_copy = scaler.transform(testX_copy)

    predY = model.predict(testX_copy)
    # print('finished predicition')
    testY_copy['pred_finishTime'] = predY

    overall = pd.merge(testX_bak, testY_copy, how='right',
                       left_index=True, right_index=True)

    overall["pred_plc"] = overall.groupby(['date', 'raceNo'])[
        "pred_finishTime"].rank()
    overall["real_plc"] = overall.groupby(['date', 'raceNo'])["plc"].rank()

    overall = overall[(overall['pred_plc'] <= 1) & (
        overall['odds'].astype(float) <= odds_max) & (overall['odds'].astype(float) >= odds_min)]
    overall.loc[overall['real_plc'] <= 3, 'real_first_3'] = 1

    overall.fillna(0, inplace=True)

    first_1 = accuracy_score(overall['real_plc'].round(
        0), overall['pred_plc'].round(0))
    first_3 = accuracy_score(overall['real_first_3'], overall['pred_plc'])
    logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
                 first_1, first_3, overall['real_plc'].count(), train_test_col)

    overall = overall.tail(20)
    first_1_recent_20 = accuracy_score(
        overall['real_plc'].round(0), overall['pred_plc'].round(0))
    first_3_recent_20 = accuracy_score(
        overall['real_first_3'], overall['pred_plc'])
    logging.info('%s, Accuracy (Recent 20) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
                 first_1_recent_20, first_3_recent_20, overall['real_plc'].count(), train_test_col)

    overall = overall.tail(10)
    first_1_recent_10 = accuracy_score(
        overall['real_plc'].round(0), overall['pred_plc'].round(0))
    first_3_recent_10 = accuracy_score(
        overall['real_first_3'], overall['pred_plc'])
    logging.info('%s, Accuracy (Recent 10) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
                 first_1_recent_10, first_3_recent_10, overall['real_plc'].count(), train_test_col)

    return overall


if __name__ == "__main__":

    train_test_col = ['B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'raceCourse_ST','Runs_6', 'Runs_5', 'Runs_4', 'Runs_3', 'Runs_2', 'Runs_1', 'TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'Draw', 'Rtg.+/-', 'AWT', 'class', 'DamRank', 'HorseMatchRank', 'Age', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)']

    train_test_col =['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'SB', 'B']

    result = test(train_test_col, 15, 5)
    # result = test(train_test_col)

    result = result[['date', 'raceNo', 'horseNo', 'plc',
                     'odds', 'pred_finishTime', 'real_plc', 'pred_plc', ]]

    logging.info('Test Result \n %s', result.tail(10))


# 1000M
# 2019-06-03 21:49:44  INFO Accuracy (All) first_1: 0.4839, first_3: 0.7097, No. of rows: 31, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Age', 'AWT']
# 2019-06-03 21: 49: 50  INFO Accuracy(All) first_1: 0.4545, first_3: 0.7273, No. of rows: 33, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Age', 'going_GOOD']
# 2019-06-03 21: 53: 06  INFO Accuracy(All) first_1: 0.3846, first_3: 0.7308, No. of rows: 26, col: ['Rtg.+/-', 'class', 'Age', 'Horse Wt. (Declaration)', 'raceCourse_ST']
# 2019-06-03 21: 53: 11  INFO Accuracy(All) first_1: 0.4815, first_3: 0.7407, No. of rows: 27, col: ['Rtg.+/-', 'class', 'Age', 'Horse Wt. (Declaration)', 'CP']
# 2019-06-03 21: 53: 39  INFO Accuracy(All) first_1: 0.4688, first_3: 0.8125, No. of rows: 32, col: ['Rtg.+/-', 'class', 'Age', 'Draw', 'raceCourse_ST']
# 2019-06-03 21: 59: 14  INFO Accuracy(All) first_1: 0.4348, first_3: 0.8261, No. of rows: 23, col: ['Rtg.+/-', 'class', 'HorseMatchRank', 'horseRank', 'P']
# 2019-06-03 23: 37: 32  INFO Accuracy(All) first_1: 0.6364, first_3: 0.8636, No. of rows: 22, col: ['Rtg.+/-', 'Age', 'raceCourse_HV', 'going_GOOD', 'E']
#
#
# 1200M
# 2019-06-04 10:22:23  INFO 1200M, Accuracy (All) first_1: 0.4625, first_3: 0.6625, No. of rows: 80, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Horse Wt. (Declaration)', 'Sex_g']
# 2019-06-04 10:22:26  INFO 1200M, Accuracy (All) first_1: 0.4578, first_3: 0.6747, No. of rows: 83, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Horse Wt. (Declaration)', 'SB']
# 2019-06-04 10:22:33  INFO 1200M, Accuracy (All) first_1: 0.4557, first_3: 0.6835, No. of rows: 79, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Horse Wt. (Declaration)', 'V']
# 2019-06-04 10:22:34  INFO 1200M, Accuracy (All) first_1: 0.4545, first_3: 0.6883, No. of rows: 77, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Horse Wt. (Declaration)', 'H']
# 2019-06-04 10:24:09  INFO 1200M, Accuracy (All) first_1: 0.4444, first_3: 0.7160, No. of rows: 81, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Draw', 'going_GOOD']
# 2019-06-04 10:13:19  INFO 1200M, Accuracy (All) first_1: 0.4699, first_3: 0.6867, No. of rows: 83, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Horse Wt. (Declaration)', 'SB']
# 2019-06-04 10:15:12  INFO 1200M, Accuracy (All) first_1: 0.4595, first_3: 0.7027, No. of rows: 74, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'raceCourse_HV', 'going_YIELDING']
# 2019-06-04 10:15:20  INFO 1200M, Accuracy (All) first_1: 0.4865, first_3: 0.6892, No. of rows: 74, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'going_YIELDING', 'BO']
# 2019-06-04 10:15:40  INFO 1200M, Accuracy (All) first_1: 0.4800, first_3: 0.7067, No. of rows: 75, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Sex_h', 'SB']
# 2019-06-04 10:15:56  INFO 1200M, Accuracy (All) first_1: 0.4730, first_3: 0.7162, No. of rows: 74, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'SB', 'B']


# 1400M
# 2019-06-03 14:23:56  INFO Accuracy (All) first_1: 0.5385, first_3: 0.8077, No. of rows: 26, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'going_YIELDING', 'Sex_c']
# 2019-06-03 14:26:36  INFO Accuracy (All) first_1: 0.3793, first_3: 0.8276, No. of rows: 29, col: ['Rtg.+/-', 'class', 'Horse Wt. (Declaration)', 'AWT', 'Runs_3']
# 2019-06-03 14:27:36  INFO Accuracy (All) first_1: 0.5600, first_3: 0.8400, No. of rows: 25, col: ['Rtg.+/-', 'class', 'Horse Wt. (Declaration)', 'going_YIELDING', 'going_GOOD TO FIRM']
# 2019-06-03 14:32:51  INFO Accuracy (All) first_1: 0.4400, first_3: 0.8400, No. of rows: 25, col: ['Rtg.+/-', 'class', 'AWT', 'SR', 'H']
# 2019-06-03 14:42:53  INFO Accuracy (All) first_1: 0.6296, first_3: 0.8519, No. of rows: 27, col: ['Rtg.+/-', 'class', 'Runs_5', 'raceCourse_ST', 'H']
# 2019-06-03 14:44:19  INFO Accuracy (All) first_1: 0.5833, first_3: 0.8750, No. of rows: 24, col: ['Rtg.+/-', 'class', 'raceCourse_ST', 'going_YIELDING', 'H']
# 2019-06-03 14:45:54  INFO Accuracy (All) first_1: 0.6364, first_3: 0.9091, No. of rows: 22, col: ['Rtg.+/-', 'class', 'going_YIELDING', 'P', 'H']
# 2019-06-03 14:46:24  INFO Accuracy (All) first_1: 0.6522, first_3: 0.9130, No. of rows: 23, col: ['Rtg.+/-', 'class', 'going_GOOD TO YIELDING', 'P', 'H']

# 1600M
# 2019-06-03 15:00:05  INFO Accuracy (All) first_1: 0.4348, first_3: 0.7391, No. of rows: 23, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'going_YIELDING', 'TT']
# 2019-06-03 15:01:28  INFO Accuracy (All) first_1: 0.3684, first_3: 0.7895, No. of rows: 19, col: ['Rtg.+/-', 'class', 'Horse Wt. (Declaration)', 'going_GOOD TO YIELDING', 'Sex_r']
# 2019-06-03 15:03:40  INFO Accuracy (All) first_1: 0.5000, first_3: 0.8125, No. of rows: 16, col: ['Rtg.+/-', 'class', 'Draw', 'going_GOOD TO FIRM', 'H']
# 2019-06-03 15:05:06  INFO Accuracy (All) first_1: 0.5294, first_3: 0.8235, No. of rows: 17, col: ['Rtg.+/-', 'class', 'SireRank', 'SR', 'CP']

# 1650M
# 2019-06-03 15:59:39  INFO Accuracy (All) first_1: 0.4000, first_3: 0.7111, No. of rows: 45, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Runs_2', 'TT']
# 2019-06-03 15:59:41  INFO Accuracy (All) first_1: 0.4222, first_3: 0.7333, No. of rows: 45, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'Runs_4', 'Runs_6']
# 2019-06-03 16:00:10  INFO Accuracy (All) first_1: 0.3864, first_3: 0.7727, No. of rows: 44, col: ['Rtg.+/-', 'class', 'Age', 'Horse Wt. (Declaration)', 'Draw']
# 2019-06-03 16:01:54  INFO Accuracy (All) first_1: 0.5556, first_3: 0.7778, No. of rows: 36, col: ['Rtg.+/-', 'class', 'Horse Wt. (Declaration)', 'going_GOOD TO FIRM', 'Sex_r']
# 2019-06-03 16:01:55  INFO Accuracy (All) first_1: 0.4762, first_3: 0.7857, No. of rows: 42, col: ['Rtg.+/-', 'class', 'Horse Wt. (Declaration)', 'going_GOOD TO FIRM', 'Sex_g']
# 2019-06-03 16:04:10  INFO Accuracy (All) first_1: 0.4286, first_3: 0.7857, No. of rows: 42, col: ['Rtg.+/-', 'class', 'AWT', 'Runs_2', 'Sex_g']
# 2019-06-03 16:08:29  INFO Accuracy (All) first_1: 0.3830, first_3: 0.7872, No. of rows: 47, col: ['Rtg.+/-', 'class', 'Runs_1', 'PC', 'XB']
# 2019-06-03 16:11:00  INFO Accuracy (All) first_1: 0.4444, first_3: 0.8056, No. of rows: 36, col: ['Rtg.+/-', 'class', 'raceCourse_ST', 'P', 'V']
# 2019-06-03 17:04:22  INFO Accuracy (All) first_1: 0.3095, first_3: 0.8095, No. of rows: 42, col: ['Rtg.+/-', 'DamRank', 'TrainerRank', 'Sex_g', 'Sex_c']
# 2019-06-03 17:04:22  INFO Accuracy (All) first_1: 0.3922, first_3: 0.8431, No. of rows: 51, col: ['Rtg.+/-', 'DamRank', 'TrainerRank', 'Sex_h', 'B']


# 1800M
# 2019-06-03 17:35:33  INFO 1800M, Accuracy (All) first_1: 0.2500, first_3: 0.8125, No. of rows: 16, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'TrainerRank', 'going_GOOD TO YIELDING']
# 2019-06-03 17:35:33  INFO 1800M, Accuracy (All) first_1: 0.3333, first_3: 0.8333, No. of rows: 18, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'TrainerRank', 'Sex_r']
# 2019-06-03 17:35:33  INFO 1800M, Accuracy (All) first_1: 0.3333, first_3: 0.8667, No. of rows: 15, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'TrainerRank', 'Sex_h']
# 2019-06-03 17:35:33  INFO 1800M, Accuracy (All) first_1: 0.3571, first_3: 0.9286, No. of rows: 14, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'TrainerRank', 'Sex_g']
# 2019-06-03 17:35:33  INFO 1800M, Accuracy (All) first_1: 0.2667, first_3: 0.9333, No. of rows: 15, col: ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'TrainerRank', 'E']
# 2019-06-03 17:37:52  INFO 1800M, Accuracy (All) first_1: 0.5000, first_3: 0.8571, No. of rows: 14, col: ['Rtg.+/-', 'class', 'Age', 'raceCourse_HV', 'Sex_r']
# 2019-06-03 17:50:26  INFO 1800M, Accuracy (All) first_1: 0.3636, first_3: 1.0000, No. of rows: 11, col: ['Rtg.+/-', 'Wt.+/- (vs Declaration)', 'horseRank', 'going_GOOD TO FIRM', 'BO']
