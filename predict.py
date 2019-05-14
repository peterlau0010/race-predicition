from joblib import dump, load
import numpy as np
import pandas as pd
import logging

date = '20190515'





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


# ========== Fill all Nan will mean ==============
X_test_backup = X_test.copy()
X_test.fillna(sireRank.mean(), inplace=True)
X_test.fillna(damRank.mean(), inplace=True)

logging.info('X_test: %s \n %s', np.shape(X_test), X_test)
    

# ========= Add missing column (Start)===================
X_test = pd.get_dummies(
    X_test, columns=['class'], prefix=['class'])

X_test = pd.get_dummies(
    X_test, columns=['road'], prefix=['road'])

X_test = pd.get_dummies(
    X_test, columns=['dist'], prefix=['dist'])

X_test = pd.get_dummies(
    X_test, columns=['going'], prefix=['going'])

X_test = pd.get_dummies(
    X_test, columns=['raceCourse'], prefix=['raceCourse'])

if 'dist_1000M' not in X_test:
    X_test['dist_1000M'] = np.NaN

if 'dist_1200M' not in X_test:
    X_test['dist_1200M'] = np.NaN

if 'dist_1000M' not in X_test:
    X_test['dist_1000M'] = np.NaN

if 'dist_1400M' not in X_test:
    X_test['dist_1400M'] = np.NaN

if 'dist_1600M' not in X_test:
    X_test['dist_1600M'] = np.NaN

if 'dist_1650M' not in X_test:
    X_test['dist_1650M'] = np.NaN

if 'dist_1800M' not in X_test:
    X_test['dist_1800M'] = np.NaN

if 'dist_2000M' not in X_test:
    X_test['dist_2000M'] = np.NaN

if 'road_ALL WEATHER TRACK' not in X_test:
    X_test['road_ALL WEATHER TRACK'] = np.NaN

if 'road_TURF - A Course' not in X_test:
    X_test['road_TURF - A Course'] = np.NaN

if 'road_TURF - B Course' not in X_test:
    X_test['road_TURF - B Course'] = np.NaN

if 'road_TURF - C Course' not in X_test:
    X_test['road_TURF - C Course'] = np.NaN

if 'going_GOOD' not in X_test:
    X_test['going_GOOD'] = np.NaN

if 'going_GOOD TO FIRM' not in X_test:
    X_test['going_GOOD TO FIRM'] = np.NaN

if 'going_GOOD TO YIELDING' not in X_test:
    X_test['going_GOOD TO YIELDING'] = np.NaN

if 'going_YIELDING' not in X_test:
    X_test['going_YIELDING'] = np.NaN

if 'going_WET SLOW' not in X_test:
    X_test['going_WET SLOW'] = np.NaN


if 'class_Class 1' not in X_test:
    X_test['class_Class 1'] = np.NaN

if 'class_Class 2' not in X_test:
    X_test['class_Class 2'] = np.NaN

if 'class_Class 3' not in X_test:
    X_test['class_Class 3'] = np.NaN

if 'class_Class 4' not in X_test:
    X_test['class_Class 4'] = np.NaN

if 'class_Class 5' not in X_test:
    X_test['class_Class 5'] = np.NaN

if 'raceCourse_HV' not in X_test:
    X_test['raceCourse_HV'] = np.NaN

if 'raceCourse_ST' not in X_test:
    X_test['raceCourse_ST'] = np.NaN

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
X_test = X_test[['draw', 'Age', 'Win%_y', 'Win%_x', 'DamRank', 'SireRank', 'awt', 'dhw', 'dist_1000M', 'dist_1200M', 'dist_1400M', 'dist_1600M', 'dist_1650M', 'dist_1800M', 'dist_2000M', 'road_ALL WEATHER TRACK', 'road_TURF - A Course', 'road_TURF - B Course',
                 'road_TURF - C Course', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_WET SLOW', 'going_YIELDING', 'class_Class 1', 'class_Class 2', 'class_Class 3', 'class_Class 4', 'class_Class 5', 'raceCourse_HV', 'raceCourse_ST']]
logging.info('X_test: %s \n %s', np.shape(X_test), X_test)


# ========== Fill all Nan to 0 ===================
X_test.fillna(0, inplace=True)

y_test = X_test.copy()

# ========== Standardization and Prediction ======
X_test = X_test.astype(float)
X_test = scalerX.transform(X_test)
y_pred = model.predict(X_test)
X_test = scalerX.inverse_transform(X_test)


# ========== Generat the result ==================
y_test.loc[:, 'pred_finish_time'] = y_pred
regression_result = pd.merge(X_test_backup, y_test, how='left', left_index=True, right_index=True)
regression_result.loc[:, 'pred_plc'] = regression_result.groupby(['raceNo'])["pred_finish_time"].rank()
regression_result = regression_result[['raceNo','Horse No.','Horse','draw','pred_finish_time','pred_plc']]
regression_result = regression_result.sort_values(by=['raceNo','pred_plc'],ascending=[True,True])

logging.info('Regression result: %s \n %s', np.shape(regression_result), regression_result)

headers = ','.join(map(str, regression_result.columns.values))
np.savetxt('./Report/regression_result_'+date+'.csv', regression_result.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')

print(regression_result[regression_result['pred_plc'] <= 1])
