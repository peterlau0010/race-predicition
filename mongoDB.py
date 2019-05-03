import pymongo as pymg
import pprint
import pandas
import numpy



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
            query = { "code": horse['Code'] }
            values = { "$set": horse.to_dict() }
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



horseCSV = pandas.read_csv('regression_report.csv', sep=',')