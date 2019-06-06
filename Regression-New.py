import Train 
import Test
import Predict
from multiprocessing import Process, Value,Pool
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import logging
import itertools
import time

# Config
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler("{0}/{1}.log".format('./Log/', 'Regression-New'))])



date = '20190605'
dist = '1650M'
min_odds = 5
max_odds = 50
col = ['Runs_2', 'Draw', 'Rtg.+/-']

""" 
No. Action
0   Train
1   Test with col
2   Predict with col
"""
action = 1



def init(arg1,arg2,):
    global first_1_max
    global first_3_max
    first_1_max = arg1
    first_3_max = arg2

def train_async(train_test_col,testX,testY,trainX,trainY,testX_bak,predX,min_odds,max_odds):
    model, scaler = Train.train(train_test_col,trainX,trainY)
    overall = Test.test(model,scaler,train_test_col,testX,testY,testX_bak,min_odds,max_odds)
    if overall is None or overall['real_plc'].count() < 15:
        return
    first_1 = accuracy_score(overall['real_plc'].round(
        0), overall['pred_plc'].round(0))
    first_3 = accuracy_score(overall['real_first_3'], overall['pred_plc'])
    with first_1_max.get_lock(), first_3_max.get_lock():
        # if (first_1 + first_3) > (first_1_max.value + first_3_max.value):
        if first_3 >= first_3_max.value or first_3 >= 75:
            logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
                 first_1, first_3, overall['real_plc'].count(), train_test_col)
            first_1_max.value = first_1
            first_3_max.value = first_3

            pred_result = Predict.predict(predX,train_test_col,model,scaler)
            pred_result = pred_result[['date', 'raceNo', 'Horse No.','Horse',
                    'pred_finishTime', 'pred_plc', ]]
            logging.info(pred_result[pred_result['pred_plc']==1])
            # logging.info('%s, Accuracy (All) first_1: %.4f, first_3: %.4f, No. of rows: %s, col: %s', testX['dist'].values[0],
            #      first_1, first_3, overall['real_plc'].count(), train_test_col)



if __name__ == "__main__":
    
    """ Parameter setup """
    testX = pd.read_csv('Processed Data/testX_'+date+'_'+dist+'.csv',
                        header=0, low_memory=False)
    testY = pd.read_csv('Processed Data/testY_'+date+'_'+dist+'.csv',
                        header=0, low_memory=False)
    trainX = pd.read_csv('Processed Data/trainX_'+date+'_'+dist+'.csv',
                        header=0, low_memory=False)
    trainY = pd.read_csv('Processed Data/trainY_'+date+'_'+dist+'.csv',
                        header=0, low_memory=False)
    predX = pd.read_csv('Processed Data/pred_'+date+'_'+dist+'.csv',
                     header=0, low_memory=False)
    testX_bak = testX.copy()

    first_1_max = Value('f', 0)
    first_3_max = Value('f', 0)

    if action == 0:
        """ Pool initital """
        pool = Pool(initializer = init, initargs = (first_1_max,first_3_max, ))
        train_test_col = ['B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','AWT','DamRank','HorseMatchRank','Horse Wt. (Declaration)','Age','Wt.+/- (vs Declaration)','class']

        perm = itertools.permutations(train_test_col,1)

        for i in perm:
            base = ['Rtg.+/-','Draw']
            i = base + list(i)
            perm2 = itertools.permutations(i,len(i))
            for i in perm2:
            # result = pool.apply_async(train_async, (base,testX,testY,trainX,trainY,testX_bak,predX))
                result = pool.apply_async(train_async, (list(i),testX,testY,trainX,trainY,testX_bak,predX,min_odds,max_odds))
        time.sleep(5)
        pool.close()
        pool.join()

    if action == 2: 
        """ Prediction """
        model, scaler = Train.train(col,trainX,trainY)
        pred_result = Predict.predict(predX,col,model,scaler)
        pred_result = pred_result[['date', 'raceNo', 'Horse No.','Horse',
                        'pred_finishTime', 'pred_plc', ]]
        logging.info('Prediction Result \n %s', pred_result[pred_result['pred_plc']<=2])

    if action == 1:
        """ Test """
        model, scaler = Train.train(col,trainX,trainY)
        test_result = Test.test(model,scaler,col,testX,testY,testX_bak,min_odds,max_odds)
        test_result = test_result.sort_values(by=['date', 'raceNo'], ascending=[False,True])
        # logging.info('Test Result \n %s', test_result.sort_values(by=['date', 'raceNo'], ascending=[False,False]).head())
        for i in range(0,30,10):
            test_result_print = test_result.head(999 if i==0 else i)
            first_1 = accuracy_score(test_result_print['real_plc'].round(0), test_result_print['pred_plc'].round(0))
            first_3 = accuracy_score(test_result_print['real_first_3'], test_result_print['pred_plc'])
            logging.info('%s, Accuracy (%s) first_1: %.4f, first_3: %.4f, col: %s', testX['dist'].values[0], test_result_print['real_plc'].count(),
                    first_1, first_3,col)
        test_result = test_result[['date', 'raceNo', 'horseNo', 'plc',
                     'odds', 'pred_finishTime', 'real_plc', 'pred_plc','real_first_3' ]]
        logging.info('Test Result \n %s', test_result.head())




# train_test_col = ['B','H','TT','CP','V','XB','SR','P','PC','E','BO','PS','SB','Sex_c','Sex_f','Sex_g','Sex_h','Sex_r','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','going_YIELDING','raceCourse_HV','raceCourse_ST','Runs_6','Runs_5','Runs_4','Runs_3','Runs_2','Runs_1','TrainerRank','SireRank','horseRank','JockeyRank','Draw','AWT','DamRank','HorseMatchRank','Horse Wt. (Declaration)','Age','Wt.+/- (vs Declaration)','class','Rtg.+/-']

""" 1200M 

# 3
2019-06-05 20: 35: 55  INFO 1200M, Accuracy(All) first_1: 0.4459, first_3: 0.6757, No. of rows: 74, col: ['Rtg.+/-', 'TT', 'Age']
2019-06-05 20: 35: 55  INFO 1200M, Accuracy(All) first_1: 0.4324, first_3: 0.6892, No. of rows: 74, col: ['Rtg.+/-', 'TT', 'Wt.+/- (vs Declaration)']
2019-06-05 20: 42: 44  INFO 1200M, Accuracy(All) first_1: 0.4416, first_3: 0.7273, No. of rows: 77, col: ['Wt.+/- (vs Declaration)', 'Rtg.+/-', 'TT']

# 4
2019-06-05 20: 53: 31  INFO 1200M, Accuracy(All) first_1: 0.4487, first_3: 0.7179, No. of rows: 78, col: ['Wt.+/- (vs Declaration)', 'Rtg.+/-', 'TT', 'Draw']
2019-06-05 20: 53: 32  INFO 1200M, Accuracy(All) first_1: 0.4722, first_3: 0.7361, No. of rows: 72, col: ['Wt.+/- (vs Declaration)', 'Rtg.+/-', 'Draw', 'TT']
2019-06-05 20: 53: 52  INFO 1200M, Accuracy(All) first_1: 0.4324, first_3: 0.7432, No. of rows: 74, col: ['Draw', 'Wt.+/- (vs Declaration)', 'TT', 'Rtg.+/-']

# 5
2019-06-05 21: 05: 51  INFO 1200M, Accuracy(All) first_1: 0.4286, first_3: 0.7403, No. of rows: 77, col: ['Draw', 'Wt.+/- (vs Declaration)', 'Rtg.+/-', 'TT', 'class']
2019-06-05 21: 05: 58  INFO 1200M, Accuracy(All) first_1: 0.4400, first_3: 0.7600, No. of rows: 75, col: ['Draw', 'TT', 'class', 'Wt.+/- (vs Declaration)', 'Rtg.+/-']
2019-06-05 21: 30: 30  INFO 1200M, Accuracy(All) first_1: 0.4286, first_3: 0.7662, No. of rows: 77, col: ['Draw', 'Rtg.+/-', 'going_YIELDING', 'TT', 'Wt.+/- (vs Declaration)']
2019-06-05 21: 31: 11  INFO 1200M, Accuracy(All) first_1: 0.4545, first_3: 0.7662, No. of rows: 77, col: ['Rtg.+/-', 'Draw', 'going_YIELDING', 'Wt.+/- (vs Declaration)', 'TT']
2019-06-05 21: 32: 12  INFO 1200M, Accuracy(All) first_1: 0.4359, first_3: 0.7692, No. of rows: 78, col: ['TT', 'Draw', 'Wt.+/- (vs Declaration)', 'going_GOOD TO YIELDING', 'Rtg.+/-']
2019-06-05 21: 37: 35  INFO 1200M, Accuracy(All) first_1: 0.4730, first_3: 0.7703, No. of rows: 74, col: ['Wt.+/- (vs Declaration)', 'Sex_h', 'Rtg.+/-', 'TT', 'Draw']
2019-06-05 21: 37: 41  INFO 1200M, Accuracy(All) first_1: 0.4487, first_3: 0.7821, No. of rows: 78, col: ['TT', 'Wt.+/- (vs Declaration)', 'Rtg.+/-', 'Sex_h', 'Draw']
2019-06-05 21: 43: 34  INFO 1200M, Accuracy(All) first_1: 0.4533, first_3: 0.7867, No. of rows: 75, col: ['Rtg.+/-', 'Draw', 'SB', 'TT', 'Wt.+/- (vs Declaration)']

# 6 
2019-06-05 22:26:22  INFO 1200M, Accuracy (All) first_1: 0.4079, first_3: 0.7500, No. of rows: 76, col: ['Rtg.+/-', 'TT', 'Draw', 'Wt.+/- (vs Declaration)', 'SB', 'class']
2019-06-05 22:43:23  INFO 1200M, Accuracy (All) first_1: 0.4691, first_3: 0.7531, No. of rows: 81, col: ['Draw', 'SB', 'Horse Wt. (Declaration)', 'Wt.+/- (vs Declaration)', 'TT', 'Rtg.+/-']
2019-06-05 22:47:20  INFO 1200M, Accuracy (All) first_1: 0.5190, first_3: 0.7595, No. of rows: 79, col: ['Wt.+/- (vs Declaration)', 'SB', 'Rtg.+/-', 'TT', 'Horse Wt. (Declaration)', 'Draw']
2019-06-05 23:05:03  INFO 1200M, Accuracy (All) first_1: 0.4533, first_3: 0.7600, No. of rows: 75, col: ['Rtg.+/-', 'Draw', 'TT', 'SB', 'AWT', 'Wt.+/- (vs Declaration)']
2019-06-05 23:06:14  INFO 1200M, Accuracy (All) first_1: 0.4521, first_3: 0.7671, No. of rows: 73, col: ['Rtg.+/-', 'AWT', 'Wt.+/- (vs Declaration)', 'Draw', 'SB', 'TT']
2019-06-05 23:12:44  INFO 1200M, Accuracy (All) first_1: 0.4557, first_3: 0.7722, No. of rows: 79, col: ['AWT', 'Wt.+/- (vs Declaration)', 'TT', 'Draw', 'Rtg.+/-', 'SB']
2019-06-06 01:00:22  INFO 1200M, Accuracy (All) first_1: 0.4400, first_3: 0.7733, No. of rows: 75, col: ['TT', 'Wt.+/- (vs Declaration)', 'SB', 'Rtg.+/-', 'going_GOOD TO YIELDING', 'Draw']
2019-06-06 01:47:28  INFO 1200M, Accuracy (All) first_1: 0.4667, first_3: 0.7733, No. of rows: 75, col: ['SB', 'Wt.+/- (vs Declaration)', 'Draw', 'Sex_f', 'TT', 'Rtg.+/-']
2019-06-06 02:10:51  INFO 1200M, Accuracy (All) first_1: 0.4400, first_3: 0.7733, No. of rows: 75, col: ['SB', 'Rtg.+/-', 'TT', 'Draw', 'BO', 'Wt.+/- (vs Declaration)']

"""


""" 1600M
Odds min=5 max=50
2019-06-06 17:53:59  INFO 1650M, Accuracy (All) first_1: 0.3846, first_3: 0.6923, No. of rows: 52, col: ['Runs_2', 'Draw', 'Rtg.+/-']

2019-06-06 17:44:47  INFO 1650M, Accuracy (All) first_1: 0.4048, first_3: 0.8095, No. of rows: 42, col: ['Draw', 'BO', 'TT', 'Rtg.+/-']
2019-06-06 17:44:27  INFO 1650M, Accuracy (All) first_1: 0.3864, first_3: 0.7500, No. of rows: 44, col: ['Draw', 'PC', 'TT', 'Rtg.+/-']
2019-06-06 17:44:19  INFO 1650M, Accuracy (All) first_1: 0.3864, first_3: 0.7273, No. of rows: 44, col: ['P', 'Draw', 'Rtg.+/-', 'TT']

"""