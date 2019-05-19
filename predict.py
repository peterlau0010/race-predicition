from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load
import numpy as np
import pandas as pd
import logging
import RaceParam as cfg

# =========== Read RaceParam config
totalMatch = cfg.param['totalMatch']
date = cfg.param['date']
road = cfg.param['road']

distlist = cfg.param['dist']
goinglist = cfg.param['going']
classeslist = cfg.param['classes']
raceCourselist = cfg.param['raceCourse']

logging.basicConfig(filename='./Log/predict.log', format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ========== Select valid data =====================
def selectAppropriateData(data,raceCourse,classes,dist,road,going):
    logging.info('SelectAppropriateData dist:  %s',dist)
    logging.info('SelectAppropriateData road:  %s',road)
    logging.info('SelectAppropriateData classes:  %s',classes)
    logging.info('SelectAppropriateData raceCourse:  %s',raceCourse)
    logging.info('SelectAppropriateData going:  %s',going)

    data = data if dist is None else data[(data['dist'] == dist)]
    data = data if road is None else data[(data['road'].str.contains(road))]
    data = data if classes is None else data[(
        data['class'] == classes)]
    data = data if raceCourse is None else data[(
        data['raceCourse'] == raceCourse)]
    data = data if going is None else data[(
        data['going'] == going)]
    return data


regression_result_report = pd.DataFrame()
for i,v in enumerate(distlist):
    # print(date, goinglist[i] , distlist[i], road, classeslist[i],raceCourselist[i])
    going = goinglist[i]
    dist = distlist[i]
    classes = classeslist[i]
    raceCourse = raceCourselist[i]





    # ========== Load Regression model and scaler ================
    filename = 'regressioin_model_' + date + going + raceCourse + dist + classes + '.sav'
    model = load(filename)

    filename = 'scaler_' + date + going + raceCourse + dist + classes + '.sav'
    scalerX = load(filename)

    # ========== Load required csv =====================
    X_test = pd.read_csv('./Processed Data/match_data_'+date+'.csv', header=0)
    sireRank = pd.read_csv('./Processed Data/sireRank.csv', sep=',')
    damRank = pd.read_csv('./Processed Data/damRank.csv', sep=',')



    X_test = selectAppropriateData(X_test,raceCourse,classes,dist,road,going)



    # ========== Fill all Nan will mean ==============
    X_test_backup = X_test.copy()
    X_test.fillna(sireRank.mean(), inplace=True)
    X_test.fillna(damRank.mean(), inplace=True)

    logging.info('X_test: %s \n %s', np.shape(X_test), X_test)


    # ========= Add missing column (Start)===================

    X_test = pd.get_dummies(
        X_test, columns=['Draw'], prefix=['draw'])

    X_test = pd.get_dummies(
        X_test, columns=['Age'], prefix=['Age'])

    # X_test = pd.get_dummies(
    #     X_test, columns=['Jockey'], prefix=['Jockey'])

    # X_test = pd.get_dummies(
    #     X_test, columns=['Trainer'], prefix=['Trainer'])

    logging.info('X_test: %s \n %s', np.shape(X_test), X_test)
    # ========= Add missing column (End)===================


    # ========== Rename columns ================
    X_test = X_test.rename(
        columns={'Win_y': 'Win%_y',
                'Win_x': 'Win%_x',
                'AWT': 'awt',
                'Draw': 'draw',
                'Horse Wt. (Declaration)': 'dhw',
                })


    # ========== Select required Column ================
    filename = 'preditParam' + date + going + raceCourse + dist + classes + '.csv'
    preditParam = pd.read_csv(filename)
    # print(preditParam['value'].values.tolist())
    requirelist = preditParam['value'].values.tolist()

    for r in requirelist:
        if r not in X_test:
            X_test[r] = np.NaN

    X_test = X_test[requirelist]

    logging.info('X_test: %s \n %s', np.shape(X_test), X_test)


    # ========== Fill all Nan to 0 ===================
    X_test.fillna(0, inplace=True)

    logging.info('X_test: %s \n %s', np.shape(X_test), X_test)
    y_test = X_test.copy()

    # ========== Standardization and Prediction ======
    X_test = X_test.astype(float)
    X_test = scalerX.transform(X_test)

    poly=PolynomialFeatures(degree=3)
    poly_x=poly.fit_transform(X_test)

    y_pred = model.predict(poly_x)
    # X_test = scalerX.inverse_transform(X_test)


    # ========== Generat the result ==================
    y_test.loc[:, 'pred_finish_time'] = y_pred
    regression_result = pd.merge(
        X_test_backup, y_test, how='left', left_index=True, right_index=True)
    regression_result.loc[:, 'pred_plc'] = regression_result.groupby(['raceNo'])[
        "pred_finish_time"].rank()
    regression_result = regression_result[[
        'raceNo', 'Horse No.', 'Horse', 'Draw', 'pred_finish_time', 'pred_plc']]
    regression_result = regression_result.sort_values(
        by=['raceNo', 'pred_plc'], ascending=[True, True])

    logging.info('Regression result: %s \n %s', np.shape(
        regression_result), regression_result)

    regression_result_report = regression_result_report.append(regression_result)

regression_result_report = regression_result_report.drop_duplicates()
regression_result_report = regression_result_report.sort_values(by=['raceNo'])

headers = ','.join(map(str, regression_result.columns.values))
np.savetxt('./Report/regression_result_'+date+'.csv', regression_result_report.round(0),
        delimiter=',', fmt='%s', header=headers, comments='')

print(regression_result_report[regression_result_report['pred_plc'] <= 1])
logging.info('regression_result_report: \n %s', regression_result_report[regression_result_report['pred_plc'] <= 2])