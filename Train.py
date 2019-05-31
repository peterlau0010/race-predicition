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
logging.basicConfig(filename='./Log/Train.log', format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Declare varaible
testX = pd.read_csv('Processed Data/testX_20190529.csv',
                    header=0, low_memory=False)
testY = pd.read_csv('Processed Data/testY_20190529.csv',
                    header=0, low_memory=False)
trainX = pd.read_csv('Processed Data/trainX_20190529.csv',
                     header=0, low_memory=False)
trainY = pd.read_csv('Processed Data/trainY_20190529.csv',
                     header=0, low_memory=False)
testX_bak = testX.copy()
first_1_max = Value('f', 0)
first_3_max = Value('f', 0)

print(','.join(map(str, trainX.columns.values)))


def train(train_test_col):
    # print('method init')
    global testX
    global testY
    global trainX
    global trainY
    global testX_bak
    global first_1_max
    global first_3_max
    train_test_col = list(train_test_col)
    # print(train_test_col)

    trainX_copy = trainX.copy()
    testX_copy = testX.copy()
    testY_copy = testY.copy()

    trainX_copy = trainX_copy[train_test_col]
    # print('set cols')
    trainX_copy = trainX_copy.astype(float)
    scaler = StandardScaler()
    scaler.fit(trainX_copy)
    trainX_copy = scaler.transform(trainX_copy)
    # print('Set model')
    model = MLPRegressor(random_state=1, solver='lbfgs')
    model.fit(trainX_copy, trainY.values.ravel())
    # print('finished set model')

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
        overall['odds'].astype(float) <= 15) & (overall['odds'].astype(float) >= 5)]
    overall.loc[overall['real_plc'] <= 3, 'real_first_3'] = 1
    overall.fillna(0, inplace=True)
    first_1 = accuracy_score(overall['real_plc'].round(
        0), overall['pred_plc'].round(0))
    first_3 = accuracy_score(overall['real_first_3'], overall['pred_plc'])
    # print('lock global variable')
    # print(overall['real_plc'].count())
    if overall['real_plc'].count() < 10:
        print('Number of rows < 10')
        return

    with first_1_max.get_lock(), first_3_max.get_lock():
        # print('Compare global variable')
        if (first_1 + first_3) > (first_1_max.value + first_3_max.value):
            print('updated')
            logging.info('Accuracy (All) first_1: %.4f, first_3: %.4f, col: %s',
                         first_1, first_3, train_test_col)
            first_1_max.value = first_1
            first_3_max.value = first_3
        else:
            print('perious:', first_1_max.value,
                  first_3_max.value, 'current: ', first_1, first_3)

    return overall


if __name__ == "__main__":
    # train_test_col = ['TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'Draw', 'Rtg.+/-', 'AWT','class', 'DamRank', 'HorseMatchRank', 'Age', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)']
    train_test_col = ['B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','Draw','Rtg.+/-','AWT','class','DamRank','HorseMatchRank','Age','Horse Wt. (Declaration)','Wt.+/- (vs Declaration)']                  
    train_test_col = train_test_col[::-1]
    # perm = itertools.permutations(train_test_col)
    perm = itertools.combinations(train_test_col,5)
    pool = Pool()

    for i in perm:
        # result = pool.apply(train, (i,))
        result = pool.apply_async(train, (i,))
        # result.wait()
        # print(result.get().head())
    time.sleep(5)
    pool.close()
    pool.join()


# logging.info('test_report: %s \n %s', np.shape(test_report), test_report)

# print('Accuracy score for 1st: %s', accuracy_score(test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
# print('Accuracy score for first 3: %s ', accuracy_score(test_report['real_first_3'], test_report['pred_plc']))

# print()

# logging.info('Accuracy score for 1st: %s', accuracy_score(
#     test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
# logging.info('Accuracy score for first 3: %s ', accuracy_score(
#     test_report['real_first_3'], test_report['pred_plc']))

# test_report = test_report.tail(20)
# print('Accuracy score for 1st recent 20 matches: %s', accuracy_score(
#     test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
# print('Accuracy score for first 3 recent 20 matches: %s ', accuracy_score(
#     test_report['real_first_3'], test_report['pred_plc']))
# print()

# logging.info('Accuracy score for 1st recent 20 matches: %s', accuracy_score(
#     test_report['real_plc'].round(0), test_report['pred_plc'].round(0)))
# logging.info('Accuracy score for first 3 recent 20 matches: %s ',
#             accuracy_score(test_report['real_first_3'], test_report['pred_plc']))
