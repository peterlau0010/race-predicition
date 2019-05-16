from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load
import numpy as np
import pandas as pd
import logging
import RaceParam as cfg


# =========== Read RaceParam config
date = cfg.param['date']
dist = cfg.param['dist']
road = cfg.param['road']
going = cfg.param['going']
classes = cfg.param['classes']
raceCourse = cfg.param['raceCourse']


logging.basicConfig(filename='./Log/predict.log', format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ========== Load Regression model and scaler ================
model = load('regressioin_model.sav')
scalerX = load('scaler.sav')

# ========== Load required csv =====================
X_test = pd.read_csv('./Processed Data/match_data_'+date+'.csv', header=0)
sireRank = pd.read_csv('./Processed Data/sireRank.csv', sep=',')
damRank = pd.read_csv('./Processed Data/damRank.csv', sep=',')

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
preditParam = pd.read_csv('preditParam.csv')
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

headers = ','.join(map(str, regression_result.columns.values))
np.savetxt('./Report/regression_result_'+date+'.csv', regression_result.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')

print(regression_result[regression_result['pred_plc'] <= 1])
