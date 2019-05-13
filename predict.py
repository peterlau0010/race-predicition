from joblib import dump, load
import numpy as np
import pandas as pd

model = load('regressioin_model.sav')
scalerX = load('scaler.sav')

# X_test = pd.read_csv('./Report/predicitValue.csv', header=0)

X_test = pd.read_csv('todayPredictValue.csv', header=0)
X_test_backup = X_test.copy()
X_test = X_test[['draw', 'Age', 'Win%_y', 'Win%_x', 'DamRank', 'SireRank', 'awt', 'dhw', 'dist_1000M', 'dist_1200M', 'dist_1400M', 'dist_1600M', 'dist_1650M', 'dist_1800M', 'dist_2000M', 'road_ALL WEATHER TRACK', 'road_TURF - A Course',
                 'road_TURF - B Course', 'road_TURF - C Course', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'class_Class 1', 'class_Class 2', 'class_Class 3', 'class_Class 4', 'class_Class 5', 'raceCource_HV', 'raceCource_ST']]
X_test.fillna(X_test.mean(), inplace=True)
X_test.fillna(0, inplace=True)

y_test = X_test.copy()
X_test = scalerX.transform(X_test)

y_pred = model.predict(X_test)
X_test = scalerX.inverse_transform(X_test)
# X_test['y_pred'] = y_pred
y_test.loc[:, 'y_pred'] = y_pred
y_test.loc[:, 'pred_plc'] = y_test["y_pred"].rank()


regression_result = pd.merge(X_test_backup, y_test, how='left',
                             left_index=True, right_index=True)
# regression_result = regression_result[[
#     'Horse No.', 'Horse', 'Brand No.', 'y_pred', 'pred_plc']]
regression_result = regression_result.sort_values(by=['pred_plc'])
headers = ','.join(map(str, regression_result.columns.values))
np.savetxt('./Report/regression_result.csv', regression_result.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')

print(regression_result[regression_result['pred_plc'] <= 3])
