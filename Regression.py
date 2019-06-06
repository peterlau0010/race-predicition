import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import logging
import itertools
from sklearn.metrics import mean_squared_error, accuracy_score
from multiprocessing import Process, Value, Lock, Pool, Manager
import time


date = '20190602'
dist = '1200M'
odds_max = 15
odds_min = 5
nCr_r = 5

test_col = None
test_one = None
nPr_r = None

test_col = ['SB', 'Rtg.+/-', 'TT', 'Draw', 'BO', 'Wt.+/- (vs Declaration)']
test_one = True
# nPr_r = 5

# Config
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler("{0}/{1}.log".format('./Log/', 'Regression'))])

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



testX = pd.read_csv('Processed Data/testX_'+date+'_'+dist+'.csv',
                    header=0, low_memory=False)
testY = pd.read_csv('Processed Data/testY_'+date+'_'+dist+'.csv',
                    header=0, low_memory=False)
trainX = pd.read_csv('Processed Data/trainX_'+date+'_'+dist+'.csv',
                     header=0, low_memory=False)
trainY = pd.read_csv('Processed Data/trainY_'+date+'_'+dist+'.csv',
                     header=0, low_memory=False)
pred = pd.read_csv('Processed Data/pred_'+date+'_'+dist+'.csv',
                     header=0, low_memory=False)

testX_bak = testX.copy()
pred_bak = pred.copy()

first_1_max = None
first_3_max = None

def init(arg1,arg2,):
    ''' store the counter for later use '''
    global first_1_max
    global first_3_max
    first_1_max = arg1
    first_3_max = arg2

def train(train_test_col, odds_max=999, odds_min=1):
    # print('method init')
    # global testX
    # global testY
    # global testX_bak
    global trainX
    global trainY
    train_test_col = list(train_test_col)

    trainX_copy = trainX.copy()
    # testX_copy = testX.copy()
    # testY_copy = testY.copy()

    trainX_copy = trainX_copy[train_test_col]
    trainX_copy = trainX_copy.astype(float)
    scaler = StandardScaler()
    scaler.fit(trainX_copy)
    trainX_copy = scaler.transform(trainX_copy)
    model = MLPRegressor(random_state=1, solver='lbfgs')
    model.fit(trainX_copy, trainY.values.ravel())

    return model, scaler

def test(model,scaler,train_test_col,odds_min=1,odds_max=999):
    global testX
    global testY
    global testX_bak
    testX_copy = testX.copy()
    testY_copy = testY.copy()
    # logging.info('testX_copy \n %s', testX_copy[(testX_copy['date']==20190529) & (testX_copy['raceNo']==1) & (testX_copy['horseNo']==3)])
    testX_copy = testX_copy[train_test_col]

    
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

    return overall

def test(result,train_test_col,odds_min=1,odds_max=999):
    global testX
    global testY
    global testX_bak
    global first_1_max
    global first_3_max
    train_test_col = list(train_test_col)
    model = result[0]
    scaler = result[1]
    testX_copy = testX.copy()
    testY_copy = testY.copy()
    # logging.info('testX_copy \n %s', testX_copy[(testX_copy['date']==20190529) & (testX_copy['raceNo']==1) & (testX_copy['horseNo']==3)])
    testX_copy = testX_copy[train_test_col]

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

    if overall['real_plc'].count() < 15:
        print('Number of rows < 15')
        return

    first_1 = accuracy_score(overall['real_plc'].round(
        0), overall['pred_plc'].round(0))
    first_3 = accuracy_score(overall['real_first_3'], overall['pred_plc'])

    with first_1_max.get_lock(), first_3_max.get_lock():
        # print('Compare global variable')
#         if (first_1 + first_3) > (first_1_max.value + first_3_max.value):
        if first_3 > first_3_max.value:
            # print('perious:', first_1_max.value,
            #       first_3_max.value, 'current: ', first_1, first_3)
            logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
                 first_1, first_3, overall['real_plc'].count(), train_test_col)
            first_1_max.value = first_1
            first_3_max.value = first_3

            overall = overall.head(20)
            first_1_recent_20 = accuracy_score(
                overall['real_plc'].round(0), overall['pred_plc'].round(0))
            first_3_recent_20 = accuracy_score(
                overall['real_first_3'], overall['pred_plc'])
            # logging.info('%s, Accuracy (Recent 20) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
            #             first_1_recent_20, first_3_recent_20, overall['real_plc'].count(), train_test_col)

            overall = overall.head(10)
            first_1_recent_10 = accuracy_score(
                overall['real_plc'].round(0), overall['pred_plc'].round(0))
            first_3_recent_10 = accuracy_score(
                overall['real_first_3'], overall['pred_plc'])
            # logging.info('%s, Accuracy (Recent 10) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
            #             first_1_recent_10, first_3_recent_10, overall['real_plc'].count(), train_test_col)


            pred_result = predict(model,scaler,train_test_col)
            pred_result = pred_result[['date', 'raceNo', 'Horse No.','Horse',
                            'pred_finishTime', 'pred_plc', ]]
            logging.info('Predict Result \n Accuracy (All) first_1: %.4f, first_3: %.4f \n Accuracy (Recent 20) first_1: %.4f, first_3: %.4f \n Accuracy (Recent 10) first_1: %.4f, first_3: %.4f \n %s',first_1, first_3 ,first_1_recent_20,first_3_recent_20,first_1_recent_10,first_3_recent_10, pred_result[pred_result['pred_plc']==1])

    return overall

def predict(model,scaler,train_test_col):
    global pred
    pred_copy = pred.copy()
    # ---- Add Missing Columns
    for r in train_test_col:
        if r not in pred_copy:
            pred_copy[r] = np.NaN

    # ---- Fill Missing Columns to null
    pred_copy.fillna(0, inplace=True)
    # logging.info('pred \n %s', pred[(pred['date']==20190529) & (pred['raceNo']==1) & (pred['Horse No.']==3)])
    pred_copy = pred_copy[train_test_col]
    
    pred_copy = pred_copy.astype(float)
    pred_copy = scaler.transform(pred_copy)
    pred_y = model.predict(pred_copy)

    pred_bak['pred_finishTime'] = pred_y
    
    pred_bak["pred_plc"] = pred_bak.groupby(['date', 'raceNo'])[
        "pred_finishTime"].rank()

    return pred_bak

from functools import partial
if __name__ == "__main__":
    logging.info('date: %s, dist: %s, odds_max: %s, odds_min: %s, nCr_r: %s, nPr_r: %s, test_col: %s, test_one: %s',date, dist,odds_max,odds_min,nCr_r,nPr_r,test_col,test_one)

    first_1_max = Value('f', 0)
    first_3_max = Value('f', 0)

    train_test_col = ['B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','Draw','AWT','DamRank','HorseMatchRank','Horse Wt. (Declaration)','Age','Wt.+/- (vs Declaration)','class','Rtg.+/-']

    
    train_test_col = train_test_col[::-1]
    if nPr_r == None:
        perm = itertools.combinations(train_test_col, nCr_r)
    else: 
        perm = itertools.permutations(train_test_col, nPr_r)

    # ----- Test 1 time
    if not test_col == None:
        train_test_col = test_col
        if test_one:
            perm = itertools.combinations(train_test_col, len(train_test_col))
        else:
            if not nPr_r == None:
                perm = itertools.permutations(train_test_col, nPr_r)
            else:
                perm = itertools.permutations(train_test_col, len(train_test_col))

    pool = Pool(initializer = init, initargs = (first_1_max,first_3_max, ))
    

    for i in perm:
        new_callback_function = partial(test, train_test_col=i,odds_min=odds_min, odds_max=odds_max)
        result = pool.apply_async(train, (i,odds_min,odds_max),callback=new_callback_function)
        # result = pool.apply(train, (i,5,15),callback=new_callback_function)
        # result.wait()
        # print(result)
    time.sleep(5)
    pool.close()
    pool.join()

#     exit()



#     # train_test_col = ['Rtg.+/-', 'class', 'Wt.+/- (vs Declaration)', 'raceCourse_HV', 'going_YIELDING']

#     model,scaler = train(train_test_col,5,15)

#     test_result = test(model,scaler,train_test_col,5,15)

#     first_1 = accuracy_score(test_result['real_plc'].round(
#         0), test_result['pred_plc'].round(0))
#     first_3 = accuracy_score(test_result['real_first_3'], test_result['pred_plc'])
#     logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
#                  first_1, first_3, test_result['real_plc'].count(), train_test_col)

#     # test_result = test_result.tail(20)
#     # first_1_recent_20 = accuracy_score(
#     #     test_result['real_plc'].round(0), test_result['pred_plc'].round(0))
#     # first_3_recent_20 = accuracy_score(
#     #     test_result['real_first_3'], test_result['pred_plc'])
#     # logging.info('%s, Accuracy (Recent 20) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
#     #              first_1_recent_20, first_3_recent_20, test_result['real_plc'].count(), train_test_col)

#     # test_result = test_result.tail(10)
#     # first_1_recent_10 = accuracy_score(
#     #     test_result['real_plc'].round(0), test_result['pred_plc'].round(0))
#     # first_3_recent_10 = accuracy_score(
#     #     test_result['real_first_3'], test_result['pred_plc'])
#     # logging.info('%s, Accuracy (Recent 10) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
#     #              first_1_recent_10, first_3_recent_10, test_result['real_plc'].count(), train_test_col)

#     test_result = test_result[['date', 'raceNo', 'horseNo', 'plc',
#                      'odds', 'pred_finishTime', 'real_plc', 'pred_plc', ]]

#     logging.info('Test Result \n %s', test_result.sort_values(by=['date', 'raceNo'], ascending=[False,False]).head())

#     # ----------- Predict

#     pred_result = predict(model,scaler,train_test_col)
#     pred_result = pred_result[['date', 'raceNo', 'Horse No.','Horse',
#                     'pred_finishTime', 'pred_plc', ]]
#     logging.info('Predict Result \n %s', pred_result[pred_result['pred_plc']==1])

    
# 2019-06-04 22: 19: 15  INFO 1200M, Accuracy(All) first_1: 0.4250, first_3: 0.7125, No. of rows: 80, col: ['Runs_6', 'Runs_5', 'Draw', 'Rtg.+/-', 'AWT']
# 2019-06-04 22: 19: 15  INFO Predict Result
# Accuracy(All) first_1: 0.4250, first_3: 0.7125
# Accuracy(Recent 20) first_1: 0.3500, first_3: 0.6500
# Accuracy(Recent 10) first_1: 0.4000, first_3: 0.5000
# date  raceNo  Horse No.           Horse  pred_finishTime  pred_plc
# 0   20190605.0       4          1  ORIENTAL ELITE     70298.405130       1.0
# 21  20190605.0       7         10     STRATHALLAN     70239.515788       1.0
# 2019-06-04 22: 21: 16  INFO 1200M, Accuracy(All) first_1: 0.4304, first_3: 0.7215, No. of rows: 79, col: ['Runs_6', 'Runs_5', 'Horse Wt. (Declaration)', 'Rtg.+/-', 'class']
# 2019-06-04 22: 21: 16  INFO Predict Result
# Accuracy(All) first_1: 0.4304, first_3: 0.7215
# Accuracy(Recent 20) first_1: 0.4000, first_3: 0.6000
# Accuracy(Recent 10) first_1: 0.5000, first_3: 0.5000
# date  raceNo  Horse No.           Horse  pred_finishTime  pred_plc
# 7   20190605.0       4          8  BLISSFUL EIGHT     70605.051624       1.0
# 19  20190605.0       7          8     ELITE PATCH     70181.709987       1.0
# 2019-06-04 22: 23: 53  INFO 1200M, Accuracy(All) first_1: 0.4211, first_3: 0.7237, No. of rows: 76, col: ['Runs_6', 'Runs_5', 'class', 'Horse Wt. (Declaration)', 'Rtg.+/-']
# 2019-06-04 22: 23: 53  INFO Predict Result
# Accuracy(All) first_1: 0.4211, first_3: 0.7237
# Accuracy(Recent 20) first_1: 0.4500, first_3: 0.6500
# Accuracy(Recent 10) first_1: 0.5000, first_3: 0.6000
# date  raceNo  Horse No.           Horse  pred_finishTime  pred_plc
# 7   20190605.0       4          8  BLISSFUL EIGHT     70603.150022       1.0
# 19  20190605.0       7          8     ELITE PATCH     70180.726205       1.0
# 2019-06-04 22: 45: 44  INFO 1200M, Accuracy(All) first_1: 0.4146, first_3: 0.7317, No. of rows: 82, col: ['Runs_6', 'Runs_3', 'class', 'Rtg.+/-', 'Draw']
# 2019-06-04 22: 45: 44  INFO Predict Result
# Accuracy(All) first_1: 0.4146, first_3: 0.7317
# Accuracy(Recent 20) first_1: 0.4000, first_3: 0.6000
# Accuracy(Recent 10) first_1: 0.4000, first_3: 0.5000
# date  raceNo  Horse No.           Horse  pred_finishTime  pred_plc
# 7   20190605.0       4          8  BLISSFUL EIGHT     70509.188993       1.0
# 19  20190605.0       7          8     ELITE PATCH     70146.877357       1.0
# 2019-06-05 00: 00: 15  INFO 1200M, Accuracy(All) first_1: 0.4762, first_3: 0.7381, No. of rows: 84, col: ['Runs_6', 'Wt.+/- (vs Declaration)', 'class', 'Horse Wt. (Declaration)', 'Rtg.+/-']
# 2019-06-05 00: 00: 15  INFO Predict Result
# Accuracy(All) first_1: 0.4762, first_3: 0.7381
# Accuracy(Recent 20) first_1: 0.5000, first_3: 0.6500
# Accuracy(Recent 10) first_1: 0.5000, first_3: 0.6000
# date  raceNo  Horse No.           Horse  pred_finishTime  pred_plc
# 7   20190605.0       4          8  BLISSFUL EIGHT     70573.619623       1.0
# 17  20190605.0       7          6  BALLISTIC KING     70188.902946       1.0
# 2019-06-05 00: 11: 55  INFO 1200M, Accuracy(All) first_1: 0.4268, first_3: 0.7439, No. of rows: 82, col: ['Runs_6', 'class', 'Rtg.+/-', 'Wt.+/- (vs Declaration)', 'Draw']
# 2019-06-05 00: 11: 55  INFO Predict Result
# Accuracy(All) first_1: 0.4268, first_3: 0.7439
# Accuracy(Recent 20) first_1: 0.3500, first_3: 0.7000
# Accuracy(Recent 10) first_1: 0.3000, first_3: 0.6000
# date  raceNo  Horse No.         Horse  pred_finishTime  pred_plc
# 10  20190605.0       4         11  CHARITYDREAM     70495.497348       1.0
# 19  20190605.0       7          8   ELITE PATCH     70170.971807       1.0
# 2019-06-05 02: 16: 03  INFO 1200M, Accuracy(All) first_1: 0.4557, first_3: 0.7595, No. of rows: 79, col: ['Runs_5', 'class', 'Horse Wt. (Declaration)', 'Rtg.+/-', 'Wt.+/- (vs Declaration)']
# 2019-06-05 02: 16: 03  INFO Predict Result
# Accuracy(All) first_1: 0.4557, first_3: 0.7595
# Accuracy(Recent 20) first_1: 0.3500, first_3: 0.6000
# Accuracy(Recent 10) first_1: 0.5000, first_3: 0.6000
# date  raceNo  Horse No.           Horse  pred_finishTime  pred_plc
# 7   20190605.0       4          8  BLISSFUL EIGHT     70549.798993       1.0
# 17  20190605.0       7          6  BALLISTIC KING     70194.682608       1.0
