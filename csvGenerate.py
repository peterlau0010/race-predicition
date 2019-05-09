import pandas as pd
import numpy as np

data = pd.read_csv('Processed Data/history_csv_merged.csv', header=0)
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
# print(data['Sire'].describe())

data = data.dropna(subset=['Sire'])


# ============ Generate sireRank.csv for Sire Rank ===============
sumOfPlc = data.groupby(['Sire'])['plc'].sum()
noOfMatch = data.groupby(['Sire'])['finishTime'].count()
winrate = sumOfPlc/noOfMatch
winrate = winrate.reset_index()
winrate.columns = ['Sire','SireRank']

headers = ','.join(map(str, winrate.columns.values))

np.savetxt('./Processed Data/sireRank.csv', winrate.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')


# ============ Generate damRank.csv for Dam Rank ===============
sumOfPlc = data.groupby(['Dam'])['plc'].sum()
noOfMatch = data.groupby(['Dam'])['finishTime'].count()
winrate = sumOfPlc/noOfMatch
winrate = winrate.reset_index()
winrate.columns = ['Dam','DamRank']

headers = ','.join(map(str, winrate.columns.values))

np.savetxt('./Processed Data/damRank.csv', winrate.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')