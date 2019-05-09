import pandas as pd
import numpy as np

# ======= Read CSV ==========
history_csv = pd.read_csv('./Raw Data/history.csv', sep=',', parse_dates=['date'])
horse_csv = pd.read_csv('./Raw Data/horse.csv', sep=',', parse_dates=['Foaling Date'])

jockeyList_csv = pd.read_csv('./Raw Data/jockeyList.csv', sep=',')
trainerList_csv = pd.read_csv('./Raw Data/trainerList.csv', sep=',')

jockey_csv_1718 = pd.read_csv('./Raw Data/jockey1718.csv', sep=',')
jockey_csv_1617 = pd.read_csv('./Raw Data/jockey1617.csv', sep=',')
jockey_csv_1516 = pd.read_csv('./Raw Data/jockey1516.csv', sep=',')

trainer_csv_1718 = pd.read_csv('./Raw Data/trainer1718.csv', sep=',')
trainer_csv_1617 = pd.read_csv('./Raw Data/trainer1617.csv', sep=',')
trainer_csv_1516 = pd.read_csv('./Raw Data/trainer1516.csv', sep=',')

sireRank_csv = pd.read_csv('./Processed Data/sireRank.csv', sep=',')
damRank_csv = pd.read_csv('./Processed Data/damRank.csv', sep=',')


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


# ========== Merge history_csv_merged and jockey_csv_yyyy Jockey ===========
date_after, date_before = pd.Timestamp(2017, 9, 1), pd.Timestamp(2018, 7, 31)
jockey_csv_1718 = pd.merge(jockey_csv_1718, jockeyList_csv, how='left',left_on=['Jockey'], right_on=['Jockey'])
history_csv_merged_1718 = history_csv_merged[(history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1718 = pd.merge(history_csv_merged_1718, jockey_csv_1718, how='left',left_on=['jockey'], right_on=['Jockey2'])

date_after, date_before = pd.Timestamp(2016, 9, 1), pd.Timestamp(2017, 7, 31)
jockey_csv_1617 = pd.merge(jockey_csv_1617, jockeyList_csv, how='left',left_on=['Jockey'], right_on=['Jockey'])
history_csv_merged_1617 = history_csv_merged[(history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1617 = pd.merge(history_csv_merged_1617, jockey_csv_1617, how='left',left_on=['jockey'], right_on=['Jockey2'])

date_after, date_before = pd.Timestamp(2015, 9, 1), pd.Timestamp(2016, 7, 31)
jockey_csv_1516 = pd.merge(jockey_csv_1516, jockeyList_csv, how='left',left_on=['Jockey'], right_on=['Jockey'])
history_csv_merged_1516 = history_csv_merged[(history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1516 = pd.merge(history_csv_merged_1516, jockey_csv_1516, how='left',left_on=['jockey'], right_on=['Jockey2'])

history_csv_merged = history_csv_merged_1718.append(history_csv_merged_1617).append(history_csv_merged_1516)


# ========== Merge history_csv_merged and trainer_csv_yyyy by Trainer ===========
date_after, date_before = pd.Timestamp(2017, 9, 1), pd.Timestamp(2018, 7, 31)
trainer_csv_1718 = pd.merge(trainer_csv_1718, trainerList_csv, how='left',left_on=['Trainer'], right_on=['Trainer'])
history_csv_merged_1718 = history_csv_merged[(history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1718 = pd.merge(history_csv_merged_1718, trainer_csv_1718, how='left',left_on=['trainer'], right_on=['Trainer2'])

date_after, date_before = pd.Timestamp(2016, 9, 1), pd.Timestamp(2017, 7, 31)
trainer_csv_1617 = pd.merge(trainer_csv_1617, trainerList_csv, how='left',left_on=['Trainer'], right_on=['Trainer'])
history_csv_merged_1617 = history_csv_merged[(history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1617 = pd.merge(history_csv_merged_1617, trainer_csv_1617, how='left',left_on=['trainer'], right_on=['Trainer2'])

date_after, date_before = pd.Timestamp(2015, 9, 1), pd.Timestamp(2016, 7, 31)
trainer_csv_1516 = pd.merge(trainer_csv_1516, trainerList_csv, how='left',left_on=['Trainer'], right_on=['Trainer'])
history_csv_merged_1516 = history_csv_merged[(history_csv_merged['date'] < date_before) & (history_csv_merged['date'] > date_after)]
history_csv_merged_1516 = pd.merge(history_csv_merged_1516, trainer_csv_1516, how='left',left_on=['trainer'], right_on=['Trainer2'])

history_csv_merged = history_csv_merged_1718.append(history_csv_merged_1617).append(history_csv_merged_1516)

history_csv_merged['Win%_y'] = history_csv_merged['Win%_y'].str.replace('%', '')
history_csv_merged['Win%_x'] = history_csv_merged['Win%_x'].str.replace('%', '')


# ========== Merge history_csv_merged and damRank_csv by Dam ===========
history_csv_merged = pd.merge(history_csv_merged, damRank_csv, how='left',
                              left_on=['Dam'], right_on=['Dam'])


# ========== Merge history_csv_merged and SireRank_csv by Sire ===========
history_csv_merged = pd.merge(history_csv_merged, sireRank_csv, how='left',
                              left_on=['Sire'], right_on=['Sire'])


# ========== Save as history_csv_merged.csv =====================
headers = ','.join(map(str, history_csv_merged.columns.values))
np.savetxt('./Processed Data/history_csv_merged.csv', history_csv_merged.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')
