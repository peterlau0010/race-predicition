import pandas as pd
import numpy as np
import logging

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


logging.basicConfig(filename='./Log/CombineData.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

# ======= Read CSV ==========
history_csv = pd.read_csv('./Raw Data/history.csv',
                          sep=',', parse_dates=['date'])
horse_csv = pd.read_csv('./Raw Data/horse.csv', sep=',',
                        parse_dates=['Foaling Date'])

# jockeyList_csv = pd.read_csv('./Raw Data/jockeyList.csv', sep=',')
# trainerList_csv = pd.read_csv('./Raw Data/trainerList.csv', sep=',')

# jockey_csv_1718 = pd.read_csv('./Raw Data/jockey1718.csv', sep=',')
# jockey_csv_1617 = pd.read_csv('./Raw Data/jockey1617.csv', sep=',')
# jockey_csv_1516 = pd.read_csv('./Raw Data/jockey1516.csv', sep=',')
jockey_csv_1819 = pd.read_csv('./Raw Data/jockey1819.csv', sep=',')

# trainer_csv_1718 = pd.read_csv('./Raw Data/trainer1718.csv', sep=',')
# trainer_csv_1617 = pd.read_csv('./Raw Data/trainer1617.csv', sep=',')
# trainer_csv_1516 = pd.read_csv('./Raw Data/trainer1516.csv', sep=',')
trainer_csv_1819 = pd.read_csv('./Raw Data/trainer1819.csv', sep=',')

# sireRank_csv = pd.read_csv('./Processed Data/sireRank.csv', sep=',')
# damRank_csv = pd.read_csv('./Processed Data/damRank.csv', sep=',')


# ======= Trim CSV ==========
# jockeyList_csv = jockeyList_csv[['Jockey','騎師','Jockey2']]
# headers = ','.join(map(str, jockeyList_csv.columns.values))
# np.savetxt('jockeyList.csv', jockeyList_csv.round(0),
#               delimiter=',', fmt='%s', header=headers, comments='')

# trainerList_csv = trainerList_csv[['Trainer','練馬師','Trainer2']]
# headers = ','.join(map(str, trainerList_csv.columns.values))
# np.savetxt('trainerList.csv', trainerList_csv.round(0),
#               delimiter=',', fmt='%s', header=headers, comments='')


# ========== Split history_csv "horseName" to "horseName" + "horseCode" ===========
split_result = history_csv["horseName"].str.split("(", n=1, expand=True)
history_csv["horseName"] = split_result[0]
history_csv["horseCode"] = split_result[1].str.replace(')', '')


# ========== Merge history_csv and horse_csv by horseCode and Code ===========
history_csv_merged = pd.merge(history_csv, horse_csv, how='left',
                              left_on=['horseCode'], right_on=['Code'])

history_csv_merged = history_csv_merged.drop(['horseCode'],axis=1)

history_csv_merged['date'] = pd.to_datetime(history_csv_merged['date'])
history_csv_merged['Foaling Date'] = pd.to_datetime(history_csv_merged['Foaling Date'])
history_csv_merged['Age'] = (history_csv_merged['date'] - history_csv_merged['Foaling Date']).astype('timedelta64[Y]')

logging.info('horse_csv Merged history  Size : %s \n %s', np.shape(history_csv_merged), history_csv_merged)

# ========== Merge history_csv_merged and jockey_csv_yyyy Jockey ===========
# =========== 17 to 18 ========
# jockey_csv_1718 = jockey_csv_1718[['Jockey', 'Win%']]
# date_after, date_before = pd.Timestamp(2017, 9, 1), pd.Timestamp(2018, 7, 31)
# jockey_csv_1718 = pd.merge(jockey_csv_1718, jockeyList_csv, how='left', left_on=[
#                            'Jockey'], right_on=['Jockey'])
# history_csv_merged_1718 = history_csv_merged[(
#     history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
# history_csv_merged_1718 = pd.merge(history_csv_merged_1718, jockey_csv_1718, how='left', left_on=[
#                                    'jockey'], right_on=['Jockey2'])

# # =========== 16 to 17 ========
# jockey_csv_1617 = jockey_csv_1617[['Jockey', 'Win%']]
# date_after, date_before = pd.Timestamp(2016, 9, 1), pd.Timestamp(2017, 7, 31)
# jockey_csv_1617 = pd.merge(jockey_csv_1617, jockeyList_csv, how='left', left_on=[
#                            'Jockey'], right_on=['Jockey'])
# history_csv_merged_1617 = history_csv_merged[(
#     history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
# history_csv_merged_1617 = pd.merge(history_csv_merged_1617, jockey_csv_1617, how='left', left_on=[
#                                    'jockey'], right_on=['Jockey2'])

# # =========== 15 to 16 ========
# jockey_csv_1516 = jockey_csv_1516[['Jockey', 'Win%']]
# date_after, date_before = pd.Timestamp(2015, 9, 1), pd.Timestamp(2016, 7, 31)
# jockey_csv_1516 = pd.merge(jockey_csv_1516, jockeyList_csv, how='left', left_on=[
#                            'Jockey'], right_on=['Jockey'])
# history_csv_merged_1516 = history_csv_merged[(
#     history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
# history_csv_merged_1516 = pd.merge(history_csv_merged_1516, jockey_csv_1516, how='left', left_on=[
#                                    'jockey'], right_on=['Jockey2'])

# =========== 18 to 19 ========
date_after, date_before = pd.Timestamp(2018, 9, 1), pd.Timestamp(2019, 7, 31)
history_csv_merged_1819 = history_csv_merged[(
    history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1819 = pd.merge(
    history_csv_merged_1819, jockey_csv_1819, how='left', left_on=['jockey'], right_on=['Jockey'])

history_csv_merged_1819 = history_csv_merged_1819.drop(['jockey'],axis=1)

history_csv_merged = history_csv_merged_1819

logging.info('Merged history  Size : %s \n %s', np.shape(history_csv_merged), history_csv_merged)

# ========== Merge history_csv_merged and trainer_csv_yyyy by Trainer ===========

# =========== 17 to 18 ========
# trainer_csv_1718 = trainer_csv_1718[['Trainer', 'Win%']]
# date_after, date_before = pd.Timestamp(2017, 9, 1), pd.Timestamp(2018, 7, 31)
# trainer_csv_1718 = pd.merge(trainer_csv_1718, trainerList_csv, how='left', left_on=[
#                             'Trainer'], right_on=['Trainer'])
# history_csv_merged_1718 = history_csv_merged[(
#     history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
# history_csv_merged_1718 = pd.merge(history_csv_merged_1718, trainer_csv_1718, how='left', left_on=[
#                                    'trainer'], right_on=['Trainer2'])

# # =========== 16 to 17 ========
# trainer_csv_1617 = trainer_csv_1617[['Trainer', 'Win%']]
# date_after, date_before = pd.Timestamp(2016, 9, 1), pd.Timestamp(2017, 7, 31)
# trainer_csv_1617 = pd.merge(trainer_csv_1617, trainerList_csv, how='left', left_on=[
#                             'Trainer'], right_on=['Trainer'])
# history_csv_merged_1617 = history_csv_merged[(
#     history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
# history_csv_merged_1617 = pd.merge(history_csv_merged_1617, trainer_csv_1617, how='left', left_on=[
#                                    'trainer'], right_on=['Trainer2'])

# # =========== 15 to 16 ========
# trainer_csv_1516 = trainer_csv_1516[['Trainer', 'Win%']]
# date_after, date_before = pd.Timestamp(2015, 9, 1), pd.Timestamp(2016, 7, 31)
# trainer_csv_1516 = pd.merge(trainer_csv_1516, trainerList_csv, how='left', left_on=[
#                             'Trainer'], right_on=['Trainer'])
# history_csv_merged_1516 = history_csv_merged[(
#     history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
# history_csv_merged_1516 = pd.merge(history_csv_merged_1516, trainer_csv_1516, how='left', left_on=[
#                                    'trainer'], right_on=['Trainer2'])

# =========== 18 to 19 ========
date_after, date_before = pd.Timestamp(2018, 9, 1), pd.Timestamp(2019, 7, 31)
history_csv_merged_1819 = history_csv_merged[(
    history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1819 = pd.merge(
    history_csv_merged_1819, trainer_csv_1819, how='left', left_on=['trainer'], right_on=['Trainer'])

history_csv_merged_1819 = history_csv_merged_1819.drop(['trainer'],axis=1)

history_csv_merged = history_csv_merged_1819

logging.info('Merged history  Size : %s \n %s', np.shape(history_csv_merged), history_csv_merged)

# ========== Merge history_csv_merged and damRank_csv by Dam ===========
# history_csv_merged = pd.merge(history_csv_merged, damRank_csv, how='left',
#                               left_on=['Dam'], right_on=['Dam'])


# ========== Merge history_csv_merged and SireRank_csv by Sire ===========
# history_csv_merged = pd.merge(history_csv_merged, sireRank_csv, how='left',
#                               left_on=['Sire'], right_on=['Sire'])


# ========== Save as history_csv_merged.csv =====================
headers = ','.join(map(str, history_csv_merged.columns.values))
np.savetxt('./Processed Data/history_csv_merged.csv', history_csv_merged.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')
