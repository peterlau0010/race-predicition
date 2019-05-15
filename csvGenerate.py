import pandas as pd
import numpy as np
import logging
import RaceParam as cfg

# raceCourse, classes, dist, road = None, None, None, None

# raceCourse = 'HV'
# classes = 'Class 4'
# dist = '1200M'
# road = 'TURF - A Course'
# going = 'GOOD'

totalMatch = cfg.param['totalMatch']
date = cfg.param['date']
dist = cfg.param['dist']
road = cfg.param['road']
going = cfg.param['going']
classes = cfg.param['classes']
raceCourse = cfg.param['raceCourse']



pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logging.basicConfig(filename='./Log/csvGenerate.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)


def selectAppropriateData(data,raceCourse,classes,dist,road,going):
    logging.info('SelectAppropriateData dist:  %s',dist)
    logging.info('SelectAppropriateData road:  %s',road)
    logging.info('SelectAppropriateData classes:  %s',classes)
    logging.info('SelectAppropriateData raceCourse:  %s',raceCourse)
    logging.info('SelectAppropriateData going:  %s',going)

    data = data if dist is None else data[(data['dist'] == dist)]
    data = data if road is None else data[(data['road'] == road)]
    data = data if classes is None else data[(
        data['class'] == classes)]
    data = data if raceCourse is None else data[(
        data['raceCourse'] == raceCourse)]
    data = data if going is None else data[(
        data['going'] == going)]
    return data


data = pd.read_csv('Processed Data/history_csv_merged.csv', header=0)
history_csv_merged = data.copy()

data = data[data.finishTime != '---']
data = data[data.plc != 'DISQ']
data.loc[data['plc'].str.contains('3 DH'), 'plc'] = '3'
data.loc[data['plc'].str.contains('2 DH'), 'plc'] = '2'
data.loc[data['plc'].str.contains('1 DH'), 'plc'] = '1'
data.loc[data['plc'].str.contains('4 DH'), 'plc'] = '4'
data.loc[data['plc'].str.contains('5 DH'), 'plc'] = '5'
data.loc[data['plc'].str.contains('6 DH'), 'plc'] = '6'
data.loc[data['plc'].str.contains('7 DH'), 'plc'] = '7'
data.loc[data['plc'].str.contains('8 DH'), 'plc'] = '8'
data.loc[data['plc'].str.contains('9 DH'), 'plc'] = '9'
data.loc[data['plc'].str.contains('10 DH'), 'plc'] = '10'
data['plc'] = data['plc'].astype(float)

data = selectAppropriateData(data,raceCourse,classes,dist,road,going)


# data = data.dropna(subset=['Sire'])


# ============ Generate sireRank.csv for Sire Rank ===============
sumOfPlc = data.groupby(['Sire'])['plc'].sum()
noOfMatch = data.groupby(['Sire'])['finishTime'].count()
sireRank = sumOfPlc/noOfMatch
sireRank = sireRank.reset_index()
sireRank.columns = ['Sire','SireRank']

headers = ','.join(map(str, sireRank.columns.values))

np.savetxt('./Processed Data/sireRank.csv', sireRank.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')


# ============ Generate damRank.csv for Dam Rank ===============
sumOfPlc = data.groupby(['Dam'])['plc'].sum()
noOfMatch = data.groupby(['Dam'])['finishTime'].count()
damRank = sumOfPlc/noOfMatch

damRank = damRank.reset_index()
damRank.columns = ['Dam','DamRank']

headers = ','.join(map(str, damRank.columns.values))

np.savetxt('./Processed Data/damRank.csv', damRank.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')



# ============= Update history_csv_merged.csv ============
# ========== Merge history_csv_merged and damRank_csv by Dam ===========
history_csv_merged = pd.merge(history_csv_merged, damRank, how='left',
                              left_on=['Dam'], right_on=['Dam'])


# ========== Merge history_csv_merged and SireRank_csv by Sire ===========
history_csv_merged = pd.merge(history_csv_merged, sireRank, how='left',
                              left_on=['Sire'], right_on=['Sire'])


logging.info('Merged history  Size : %s \n %s', np.shape(history_csv_merged), history_csv_merged)

# ========== Save as history_csv_merged.csv =====================
headers = ','.join(map(str, history_csv_merged.columns.values))
np.savetxt('./Processed Data/history_csv_merged_with_Sire_Dam.csv', history_csv_merged.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')