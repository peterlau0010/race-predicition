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

date = '20190602'
dist = '1200M'


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
        # if (first_1 + first_3) > (first_1_max.value + first_3_max.value):
        if first_3 > first_3_max.value:
            print('perious:', first_1_max.value,
                  first_3_max.value, 'current: ', first_1, first_3)
            logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
                 first_1, first_3, overall['real_plc'].count(), train_test_col)
            first_1_max.value = first_1
            first_3_max.value = first_3
        # else:
            # print('perious:', first_1_max.value,
            #       first_3_max.value, 'current: ', first_1, first_3)

    return overall

testX = pd.read_csv('Processed Data/testX_'+date+'_'+dist+'.csv',
                    header=0, low_memory=False)
testY = pd.read_csv('Processed Data/testY_'+date+'_'+dist+'.csv',
                    header=0, low_memory=False)
trainX = pd.read_csv('Processed Data/trainX_'+date+'_'+dist+'.csv',
                     header=0, low_memory=False)
trainY = pd.read_csv('Processed Data/trainY_'+date+'_'+dist+'.csv',
                     header=0, low_memory=False)

testX_bak = testX.copy()

first_1_max = None
first_3_max = None

def init(arg1,arg2,):
    ''' store the counter for later use '''
    global first_1_max
    global first_3_max
    first_1_max = arg1
    first_3_max = arg2

if __name__ == "__main__":
    # Declare varaible
    # global first_1_max
    # global first_3_max

    first_1_max = Value('f', 0)
    first_3_max = Value('f', 0)

    print(','.join(map(str, trainX.columns.values)))

    # train_test_col = ['TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'Draw', 'Rtg.+/-', 'AWT','class', 'DamRank', 'HorseMatchRank', 'Age', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)']
    # train_test_col = ['B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','Draw','AWT','DamRank','HorseMatchRank','Horse Wt. (Declaration)','Age','Wt.+/- (vs Declaration)','class','Rtg.+/-']

    train_test_col = ['B','H','CP','V','XB','SR','P','PC','E','BO','PS','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','AWT','DamRank','HorseMatchRank','Horse Wt. (Declaration)','Age','class',]

    train_test_col = train_test_col[::-1]
    base = ['Rtg.+/-', 'Draw', 'SB', 'TT', 'Wt.+/- (vs Declaration)']
    # perm = itertools.permutations(train_test_col)
    perm = itertools.permutations(train_test_col,1)
    pool = Pool(initializer = init, initargs = (first_1_max,first_3_max, ))

    for i in perm:
        i = base +list(i)
        # result = pool.apply_async(train, (i,))
        perm2 = itertools.permutations(i, len(i))
        for j in perm2:
        #     # result = pool.apply(train, (i,))
        #     print(j)
            result = pool.apply_async(train, (j,))
        # result.wait()
        # print(result.get().head())
    time.sleep(5)
    pool.close()
    pool.join()

# 2019-06-05 17:52:58  INFO 1200M, Accuracy (All) first_1: 0.4627, first_3: 0.6567, No. of rows: 67, col: ['Rtg.+/-']
# 2019-06-05 17:53:25  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'class']
# 2019-06-05 17:53:36  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'raceCourse_HV']
# 2019-06-05 17:53:39  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'going_YIELDING']
# 2019-06-05 17:53:39  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'going_GOOD TO FIRM']
# 2019-06-05 17:53:39  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'going_GOOD']
# 2019-06-05 17:53:40  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'Sex_r']
# 2019-06-05 17:53:40  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'Sex_f']
# 2019-06-05 17:53:42  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'PS']
# 2019-06-05 17:53:42  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'Sex_c']
# 2019-06-05 17:53:43  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'SB']
# 2019-06-05 17:53:44  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'BO']
# 2019-06-05 17:53:45  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'E']
# 2019-06-05 17:53:47  INFO 1200M, Accuracy (All) first_1: 0.4697, first_3: 0.6515, No. of rows: 66, col: ['Rtg.+/-', 'V']
# 2019-06-05 17:53:49  INFO 1200M, Accuracy (All) first_1: 0.4783, first_3: 0.6522, No. of rows: 69, col: ['Rtg.+/-', 'TT']

# 2019-06-05 20: 35: 55  INFO 1200M, Accuracy(All) first_1: 0.4459, first_3: 0.6757, No. of rows: 74, col: ['Rtg.+/-', 'TT', 'Age']
# 2019-06-05 20: 35: 55  INFO 1200M, Accuracy(All) first_1: 0.4324, first_3: 0.6892, No. of rows: 74, col: ['Rtg.+/-', 'TT', 'Wt.+/- (vs Declaration)']
# 2019-06-05 20: 42: 44  INFO 1200M, Accuracy(All) first_1: 0.4416, first_3: 0.7273, No. of rows: 77, col: ['Wt.+/- (vs Declaration)', 'Rtg.+/-', 'TT']
# 2019-06-05 20: 53: 31  INFO 1200M, Accuracy(All) first_1: 0.4487, first_3: 0.7179, No. of rows: 78, col: ['Wt.+/- (vs Declaration)', 'Rtg.+/-', 'TT', 'Draw']
# 2019-06-05 20: 53: 32  INFO 1200M, Accuracy(All) first_1: 0.4722, first_3: 0.7361, No. of rows: 72, col: ['Wt.+/- (vs Declaration)', 'Rtg.+/-', 'Draw', 'TT']
# 2019-06-05 20: 53: 52  INFO 1200M, Accuracy(All) first_1: 0.4324, first_3: 0.7432, No. of rows: 74, col: ['Draw', 'Wt.+/- (vs Declaration)', 'TT', 'Rtg.+/-']


# 2019-06-05 21: 05: 51  INFO 1200M, Accuracy(All) first_1: 0.4286, first_3: 0.7403, No. of rows: 77, col: ['Draw', 'Wt.+/- (vs Declaration)', 'Rtg.+/-', 'TT', 'class']
# 2019-06-05 21: 05: 58  INFO 1200M, Accuracy(All) first_1: 0.4400, first_3: 0.7600, No. of rows: 75, col: ['Draw', 'TT', 'class', 'Wt.+/- (vs Declaration)', 'Rtg.+/-']
# 2019-06-05 21: 30: 30  INFO 1200M, Accuracy(All) first_1: 0.4286, first_3: 0.7662, No. of rows: 77, col: ['Draw', 'Rtg.+/-', 'going_YIELDING', 'TT', 'Wt.+/- (vs Declaration)']
# 2019-06-05 21: 31: 11  INFO 1200M, Accuracy(All) first_1: 0.4545, first_3: 0.7662, No. of rows: 77, col: ['Rtg.+/-', 'Draw', 'going_YIELDING', 'Wt.+/- (vs Declaration)', 'TT']
# 2019-06-05 21: 32: 12  INFO 1200M, Accuracy(All) first_1: 0.4359, first_3: 0.7692, No. of rows: 78, col: ['TT', 'Draw', 'Wt.+/- (vs Declaration)', 'going_GOOD TO YIELDING', 'Rtg.+/-']
# 2019-06-05 21: 37: 35  INFO 1200M, Accuracy(All) first_1: 0.4730, first_3: 0.7703, No. of rows: 74, col: ['Wt.+/- (vs Declaration)', 'Sex_h', 'Rtg.+/-', 'TT', 'Draw']
# 2019-06-05 21: 37: 41  INFO 1200M, Accuracy(All) first_1: 0.4487, first_3: 0.7821, No. of rows: 78, col: ['TT', 'Wt.+/- (vs Declaration)', 'Rtg.+/-', 'Sex_h', 'Draw']
# 2019-06-05 21: 43: 34  INFO 1200M, Accuracy(All) first_1: 0.4533, first_3: 0.7867, No. of rows: 75, col: ['Rtg.+/-', 'Draw', 'SB', 'TT', 'Wt.+/- (vs Declaration)']


# 2019-06-05 22:26:22  INFO 1200M, Accuracy (All) first_1: 0.4079, first_3: 0.7500, No. of rows: 76, col: ['Rtg.+/-', 'TT', 'Draw', 'Wt.+/- (vs Declaration)', 'SB', 'class']
# 2019-06-05 22:43:23  INFO 1200M, Accuracy (All) first_1: 0.4691, first_3: 0.7531, No. of rows: 81, col: ['Draw', 'SB', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)', 'TT', 'Rtg.+/-']
# 2019-06-05 22:47:20  INFO 1200M, Accuracy (All) first_1: 0.5190, first_3: 0.7595, No. of rows: 79, col: ['Wt.+/- (vs Declaration)', 'SB', 'Rtg.+/-', 'TT', 'Horse Wt. (Declaration)', 'Draw']
# 2019-06-05 23:05:03  INFO 1200M, Accuracy (All) first_1: 0.4533, first_3: 0.7600, No. of rows: 75, col: ['Rtg.+/-', 'Draw', 'TT', 'SB', 'AWT', 'Wt.+/- (vs Declaration)']
# 2019-06-05 23:06:14  INFO 1200M, Accuracy (All) first_1: 0.4521, first_3: 0.7671, No. of rows: 73, col: ['Rtg.+/-', 'AWT', 'Wt.+/- (vs Declaration)', 'Draw', 'SB', 'TT']
# 2019-06-05 23:12:44  INFO 1200M, Accuracy (All) first_1: 0.4557, first_3: 0.7722, No. of rows: 79, col: ['AWT', 'Wt.+/- (vs Declaration)', 'TT', 'Draw', 'Rtg.+/-', 'SB']
# 2019-06-06 01:00:22  INFO 1200M, Accuracy (All) first_1: 0.4400, first_3: 0.7733, No. of rows: 75, col: ['TT', 'Wt.+/- (vs Declaration)', 'SB', 'Rtg.+/-', 'going_GOOD TO YIELDING', 'Draw']
# 2019-06-06 01:47:28  INFO 1200M, Accuracy (All) first_1: 0.4667, first_3: 0.7733, No. of rows: 75, col: ['SB', 'Wt.+/- (vs Declaration)', 'Draw', 'Sex_f', 'TT', 'Rtg.+/-']
# 2019-06-06 02:10:51  INFO 1200M, Accuracy (All) first_1: 0.4400, first_3: 0.7733, No. of rows: 75, col: ['SB', 'Rtg.+/-', 'TT', 'Draw', 'BO', 'Wt.+/- (vs Declaration)']