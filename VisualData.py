
from multiprocessing import Process, Value, Lock, Pool, Manager
import itertools
import time
import pandas as pd

match_data_race_card_bak = pd.read_csv('Raw Data/match_data_race_card_bak.csv', header=0)

match_data_race_card_bak = match_data_race_card_bak[match_data_race_card_bak['date']> 20180831]
print(match_data_race_card_bak.groupby(['dist'])['date'].nunique())

dist = ['1000M','1200M','1400M','1600M','1650M','1800M','2000M','2200M','2400M']
data = pd.DataFrame()
for d in dist:
    data = data.append(match_data_race_card_bak[match_data_race_card_bak['dist']==d][['dist','date']].head(1))
    # print(match_data_race_card_bak[match_data_race_card_bak['dist']==d][['dist','date']].head(1))

print(data)
# 1000M    60
# 1200M    76
# 1400M    39
# 1600M    31
# 1650M    48
# 1800M    52
# 2000M    13
# 2200M     8
# 2400M     3