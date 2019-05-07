import pymongo as pymg
import pprint
import pandas
import numpy
# import datetime


class MongoDB:
    # MongoDB Config
    client = pymg.MongoClient('localhost', 27017)
    db = client['HKJC']
    histroyCollection = db['history']

    def updatehistory(self):
        # CSV Config
        horseCSV = pandas.read_csv('horse.csv', sep=',')
        horseCSV = pandas.DataFrame(horseCSV)

        # horse = horseCSV.iloc[1]
        for index, horse in horseCSV.iterrows():
            horse['Age'] = int(horse['Age'])
            query = {"code": horse['Code']}
            values = {"$set": horse.to_dict()}
            x = self.histroyCollection.update_many(query, values)

    def queryhistory(self):
        history = self.histroyCollection.find({"Code": {"$ne": None}})
        print(history.count())
        return history


db = MongoDB()

# ===========Add data to DB===================
# db.updatehistory()
# history = pandas.DataFrame(list(db.queryhistory()))

# headers = ','.join(map(str, history.columns.values))
# numpy.savetxt('history_adv.csv', history, delimiter=',', fmt='%s',header=headers)



# ============== Read CSV file ================
history_adv = pandas.read_csv('history_adv.csv', sep=',', parse_dates=['date'])

trainerlist = pandas.read_csv('trainerlist_bak.csv', sep=',')
trainerlist = trainerlist[['# Trainer','練馬師','Trainer2']]
trainer1516 = pandas.read_csv('trainer1516.csv', sep=',')
trainer1617 = pandas.read_csv('trainer1617.csv', sep=',')
trainer1718 = pandas.read_csv('trainer1718.csv', sep=',')

jockeylist = pandas.read_csv('jockeylist_bak.csv', sep=',')
jockeylist = jockeylist[['# Jockey','騎師','Jockey2']]
jockey1516 = pandas.read_csv('jockey1516.csv', sep=',')
jockey1617 = pandas.read_csv('jockey1617.csv', sep=',')
jockey1718 = pandas.read_csv('jockey1718.csv', sep=',')



# ============== Create Jockey List csv =============
# jockeyList = history_adv['jockey'].value_counts()
# jockeyList = jockeyList.rename_axis('jockey').reset_index(name='counts')
# print(jockeyList)
# numpy.savetxt('JockeyList.csv', jockeyList.round(0),
#               delimiter=',', fmt='%s')

# df_out = pandas.merge(Jockey1516, jockeyList, how='right',
#                       left_on=['Jockey2'], right_on=['jockey'])

# headers = ','.join(map(str, df_out.columns.values))

# numpy.savetxt('JockeyList.csv', df_out.round(0),
#               delimiter=',', fmt='%s', header=headers)


# =============== Create trainer List csv ==============
# trainerList = history_adv['trainer'].value_counts()
# trainerList = trainerList.rename_axis('trainer').reset_index(name='counts')
# print(trainerList)
# numpy.savetxt('TrainerList.csv', trainerList.round(0),
#               delimiter=',', fmt='%s')

# df_out = pandas.merge(Trainer1516, trainerList, how='right',
#                       left_on=['Trainer2'], right_on=['trainer'])

# headers = ','.join(map(str, df_out.columns.values))
# numpy.savetxt('TrainerList.csv', df_out.round(0),
#               delimiter=',', fmt='%s', header=headers)


def generatereport(history_adv,trainerlist,jockeylist,trainer_year,jockey_year,year_from,year_to):
    date_after, date_before = pandas.Timestamp(
        year_from, 9, 1), pandas.Timestamp(year_to, 7, 31)
    history_adv = history_adv[(history_adv['date'] < date_before) & (
        history_adv['date'] > date_after)]

    trainerlist = pandas.merge(trainer_year, trainerlist, how='left',
                        left_on=['Trainer'], right_on=['# Trainer'])

    df_out = pandas.merge(history_adv, trainerlist, how='left',
                        left_on=['trainer'], right_on=['Trainer2'])

    jockeylist = pandas.merge(jockey_year, jockeylist, how='left',
                        left_on=['Jockey'], right_on=['# Jockey'])

    df_out = pandas.merge(df_out, jockeylist, how='left',
                          left_on=['jockey'], right_on=['Jockey2'])

    print(df_out.head())
    return df_out

df_out = generatereport(history_adv,trainerlist,jockeylist,trainer1617,jockey1617,2016,2017)
df_out = df_out.append(generatereport(history_adv,trainerlist,jockeylist,trainer1516,jockey1516,2015,2016))
df_out = df_out.append(generatereport(history_adv,trainerlist,jockeylist,trainer1718,jockey1718,2017,2018))

headers = ','.join(map(str, df_out.columns.values))
numpy.savetxt('history_adv15161718.csv', df_out.round(0),
              delimiter=',', fmt='%s', header=headers)
