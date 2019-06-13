import Train
import Test
import Predict
from multiprocessing import Process, Value, Pool
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import logging
import itertools
import time

# Config
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("{0}/{1}.log".format('./Log/', 'Regression-New'))])

""" 
No. Action
0   Train
1   Test with col
2   Predict with col
"""
action = 1
date = '20190608'
dist = '1400M'
min_odds = 5
max_odds = 20
top_pred_plc = 2
col = ['SR', 'Season Stakes', 'class', 'Rtg.+/-', 'BO']
base = ['SR', 'Rtg.+/-', 'going_GOOD TO YIELDING']


def init(arg1, arg2, ):
    global first_1_max
    global first_3_max
    first_1_max = arg1
    first_3_max = arg2


def train_async(train_test_col, testX, testY, trainX, trainY, testX_bak, predX, min_odds, max_odds):
    model, scaler = Train.train(train_test_col, trainX, trainY)
    overall = Test.test(model, scaler, train_test_col, testX, testY, testX_bak, min_odds, max_odds)
    if overall is None or overall['real_plc'].count() < 15:
        return
    overall_bak = overall.copy()
    overall_bak = overall_bak.sort_values(by=['date', 'raceNo'], ascending=[False, True])

    # overall = overall.tail(30)

    first_1 = accuracy_score(overall['real_plc'].round(
        0), overall['pred_plc'].round(0))
    first_3 = accuracy_score(overall['real_first_3'], overall['pred_plc'])

    with first_1_max.get_lock(), first_3_max.get_lock():
        # if (first_1 + first_3) > (first_1_max.value + first_3_max.value):

        # if first_3 >= first_3_max.value :
        if first_1 >= first_1_max.value:
            # logging.info('%s, Accuracy (%s) plc_1: %.4f, plc_1_3: %.4f, odds: %s - %s, col: %s', testX['dist'].values[0],overall['real_plc'].count(),
            #      first_1, first_3, min_odds, max_odds,  train_test_col)

            for i in range(0, 40, 10):
                test_result_print = overall_bak.head(999 if i == 0 else i)
                first_1_print = accuracy_score(test_result_print['real_plc'].round(0),
                                               test_result_print['pred_plc'].round(0))
                first_3_print = accuracy_score(test_result_print['real_first_3'], test_result_print['pred_plc'])
                logging.info('%s, Accuracy (%s) first_1: %.4f, first_3: %.4f, col: %s from: %s',
                             testX['dist'].values[0], test_result_print['real_plc'].count(),
                             first_1_print, first_3_print, train_test_col, test_result_print['date'].values[-1])

            first_1_max.value = first_1
            first_3_max.value = first_3

            pred_result = Predict.predict(predX, train_test_col, model, scaler)
            pred_result = pred_result[['date', 'raceNo', 'Horse No.', 'Horse',
                                       'pred_finishTime', 'pred_plc', ]]
            logging.info(pred_result[pred_result['pred_plc'] <= top_pred_plc])
            # logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
            #      first_1, first_3, overall['real_plc'].count(), train_test_col)


if __name__ == "__main__":

    """ Parameter setup """
    testX = pd.read_csv('Processed Data/testX_' + date + '_' + dist + '.csv',
                        header=0, low_memory=False)
    testY = pd.read_csv('Processed Data/testY_' + date + '_' + dist + '.csv',
                        header=0, low_memory=False)
    trainX = pd.read_csv('Processed Data/trainX_' + date + '_' + dist + '.csv',
                         header=0, low_memory=False)
    trainY = pd.read_csv('Processed Data/trainY_' + date + '_' + dist + '.csv',
                         header=0, low_memory=False)
    predX = pd.read_csv('Processed Data/pred_' + date + '_' + dist + '.csv',
                        header=0, low_memory=False)
    testX_bak = testX.copy()

    first_1_max = Value('f', 0)
    first_3_max = Value('f', 0)

    if action == 0:
        """ Pool initital """
        pool = Pool(initializer=init, initargs=(first_1_max, first_3_max,))
        # train_test_col = ['Draw','Rtg.','Rtg.+/-','Horse Wt. (Declaration)','Wt.+/- (vs Declaration)','Age','Season Stakes','AWT','class','Runs_1','Runs_2','Runs_3','Runs_4','Runs_5','Runs_6','B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_m','Sex_r','going_FAST','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_WET SLOW','going_YIELDING','raceCourse_HV','raceCourse_ST','SireRank','DamRank','horseRank','HorseMatchRank','JockeyRank','TrainerRank']
        train_test_col = ['Draw', 'Rtg.', 'Rtg.+/-', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)', 'Age',
                          'Season Stakes', 'AWT', 'class', 'Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'Runs_5', 'Runs_6',
                          'B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f',
                          'Sex_g', 'Sex_h', 'Sex_m', 'Sex_r', 'going_FAST', 'going_GOOD', 'going_GOOD TO FIRM',
                          'going_GOOD TO YIELDING', 'going_WET SLOW', 'going_YIELDING', 'raceCourse_HV',
                          'raceCourse_ST', 'SireRank', 'DamRank', 'horseRank', 'HorseMatchRank', 'JockeyRank',
                          'TrainerRank']

        train_test_col = [x for x in train_test_col if x not in base]
        perm = itertools.permutations(train_test_col, 1)

        for i in perm:
            # base = ['Rtg.+/-', 'H']
            i = base + list(i)
            perm2 = itertools.permutations(i, len(i))
            for i in perm2:
                # result = pool.apply_async(train_async, (base,testX,testY,trainX,trainY,testX_bak,predX))
                result = pool.apply_async(train_async,
                                          (list(i), testX, testY, trainX, trainY, testX_bak, predX, min_odds, max_odds))
        time.sleep(5)
        pool.close()
        pool.join()

    if action == 2:
        """ Prediction """
        model, scaler = Train.train(col, trainX, trainY)
        pred_result = Predict.predict(predX, col, model, scaler)
        pred_result = pred_result[['date', 'raceNo', 'Horse No.', 'Horse',
                                   'pred_finishTime', 'pred_plc', ]]
        logging.info('Prediction Result \n %s', pred_result[pred_result['pred_plc'] <= top_pred_plc])

    if action == 1:
        """ Test """
        model, scaler = Train.train(col, trainX, trainY)
        test_result = Test.test(model, scaler, col, testX, testY, testX_bak, min_odds, max_odds)
        test_result = test_result.sort_values(by=['date', 'raceNo'], ascending=[False, True])
        # logging.info('Test Result \n %s', test_result.sort_values(by=['date', 'raceNo'], ascending=[False,False]).head())
        for i in range(0, 40, 10):
            test_result_print = test_result.head(999 if i == 0 else i)
            first_1 = accuracy_score(test_result_print['real_plc'].round(0), test_result_print['pred_plc'].round(0))
            first_3 = accuracy_score(test_result_print['real_first_3'], test_result_print['pred_plc'])
            logging.info('%s, Accuracy (%s) first_1: %.4f, first_3: %.4f, col: %s from: %s', testX['dist'].values[0],
                         test_result_print['real_plc'].count(),
                         first_1, first_3, col, test_result_print['date'].values[-1])
        test_result = test_result[['date', 'raceNo', 'horseNo', 'plc',
                                   'odds', 'pred_finishTime', 'real_plc', 'pred_plc', 'real_first_3']]
        logging.info('Test Result \n %s', test_result.head(10))

    logging.info('Done')

# train_test_col = ['B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','Draw','AWT','DamRank','HorseMatchRank','Horse Wt. (Declaration)','Age','Wt.+/- (vs Declaration)','class','Rtg.+/-']
# train_test_col = ['Over Wt.','Draw','Rtg.','Rtg.+/-','Horse Wt. (Declaration)','Wt.+/- (vs Declaration)','Age','Season Stakes','AWT','class','Runs_1','Runs_2','Runs_3','Runs_4','Runs_5','Runs_6','B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_m','Sex_r','going_FAST','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_WET SLOW','going_YIELDING','raceCourse_HV','raceCourse_ST','SireRank','DamRank','horseRank','HorseMatchRank','JockeyRank','TrainerRank']

""" 1200M 
# odds: 5 ~ 15

# 2 features
2A) 2019-06-07 20:36:55  INFO 1200M, Accuracy (124) first_1: 0.3629, first_3: 0.6371, col: ['Draw', 'Rtg.+/-']
2B) 2019-06-07 20:34:54  INFO 1200M, Accuracy (129) first_1: 0.3566, first_3: 0.6512, col: ['AWT', 'Rtg.+/-']

# 3 features 
2019-06-07 21:34:03  INFO 1200M, Accuracy (30) first_1: 0.6000, first_3: 0.8667, col: ['Rtg.+/-', 'TT', 'going_GOOD']

# 4 features
2019-06-07 23:25:31  INFO 1200M, Accuracy (30) first_1: 0.6000, first_3: 0.9333, col: ['XB', 'TT', 'Rtg.+/-', 'going_GOOD'] from: 20190130.0
2019-06-07 23:41:34  INFO 1200M, Accuracy (30) first_1: 0.6000, first_3: 0.9000, col: ['class', 'TT', 'Rtg.+/-', 'going_GOOD'] from: 20190130.0

# 5 features
2019-06-08 00:39:28  INFO 1200M, Accuracy (30) plc_1: 0.5667, plc_1_3: 0.9333, odds: 5 - 15, col: ['Rtg.+/-', 'XB', 'going_GOOD', 'TT', 'P']

"""

""" 1400M
# odd 5 ~ 15

# 3 features
2019-06-08 07:25:11  INFO 1400M, Accuracy (30) plc_1: 0.5667, plc_1_3: 0.8667, odds: 5 - 15, col: ['going_GOOD', 'Rtg.+/-', 'H']

# 4 features
2019-06-08 07:38:34  INFO 1400M, Accuracy (30) plc_1: 0.5667, plc_1_3: 0.8667, odds: 5 - 15, col: ['H', 'Rtg.+/-', 'TT', 'going_GOOD']
2019-06-08 07:35:18  INFO 1400M, Accuracy (30) plc_1: 0.5333, plc_1_3: 0.8333, odds: 5 - 15, col: ['H', 'Rtg.+/-', 'going_GOOD', 'Horse Wt. (Declaration)']
2019-06-08 07:57:50  INFO 1400M, Accuracy (30) first_1: 0.5667, first_3: 0.8333, col: ['going_GOOD', 'Rtg.+/-', 'class', 'H'] from: 20180610.0


# 5 features
2019-06-08 08:41:29  INFO 1400M, Accuracy (30) plc_1: 0.5333, plc_1_3: 0.9000, odds: 5 - 15, col: ['TT', 'going_GOOD TO FIRM', 'H', 'going_GOOD', 'Rtg.+/-']
2019-06-08 08:25:29  INFO 1400M, Accuracy (30) plc_1: 0.6000, plc_1_3: 0.9000, odds: 5 - 15, col: ['going_GOOD', 'P', 'H', 'Rtg.+/-', 'TT']
2019-06-08 08:23:19  INFO 1400M, Accuracy (30) plc_1: 0.5333, plc_1_3: 0.8667, odds: 5 - 15, col: ['H', 'Rtg.+/-', 'TT', 'going_GOOD', 'SR']
"""

""" 1600M
Odds min=5 max=50
2019-06-06 17:53:59  INFO 1650M, Accuracy (All) first_1: 0.3846, first_3: 0.6923, No. of rows: 52, col: ['Runs_2', 'Draw', 'Rtg.+/-']

2019-06-06 17:44:47  INFO 1650M, Accuracy (All) first_1: 0.4048, first_3: 0.8095, No. of rows: 42, col: ['Draw', 'BO', 'TT', 'Rtg.+/-']
2019-06-06 17:44:27  INFO 1650M, Accuracy (All) first_1: 0.3864, first_3: 0.7500, No. of rows: 44, col: ['Draw', 'PC', 'TT', 'Rtg.+/-']
2019-06-06 17:44:19  INFO 1650M, Accuracy (All) first_1: 0.3864, first_3: 0.7273, No. of rows: 44, col: ['P', 'Draw', 'Rtg.+/-', 'TT']

"""
