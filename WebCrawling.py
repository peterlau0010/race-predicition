from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
import bs4
import logging
import time
import re
import numpy
import pandas


class WebCrawling:
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    def __init__(self):
        self.browser = webdriver.Safari()

    def close(self):
        self.browser.quit()

    def request(self, url):
        self.browser.get(url)
        time.sleep(5)
        html_source = bs4.BeautifulSoup(self.browser.page_source, 'lxml')
        return html_source

    def reFormatText(self, text):
        # Trim space, remove line break, remove double space
        return re.sub(' +', ' ', text.replace('\n', ' ').replace('\r', '').replace(',', '').strip())

    def getHistory(self, day, racecource, raceno):
        url_base = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?'
        url_raceDate = 'RaceDate=' + day
        url_raceCourse = '&Racecourse=' + racecource
        url_raceNo = '&RaceNo=' + raceno
        url = url_base + url_raceDate + url_raceCourse + url_raceNo

        logging.info('Access Url: %s', url)

        html_source = self.request(url)

        table = html_source.find(
            name='table', class_='f_tac table_bd draggable')

        matchDetail = html_source.find(name='tbody', class_='f_fs13')

        # Initital history list
        history = []
        history_header = []

        # Add Index value to history
        history_header.append(day)
        history_header.append(racecource)
        history_header.append(raceno)

        # Add Match Detail to history
        for t in matchDetail.find_all(name='td'):
            # logging.debug( self.reFormatText(t.text))
            history_header.append(self.reFormatText(t.text))

        # Reformat Match Detail
        history_header = history_header[0:3] + history_header[6:13]
        history_header.append(history_header[3].split(' - ')[0])
        history_header.append(history_header[3].split(' - ')[1])
        del history_header[3]
        del history_header[3]
        del history_header[5]
        # logging.info(history_header)

        for tr in table.find_all(name='tr'):
            history += history_header
            for td in tr.find_all(name='td'):
                history.append(self.reFormatText(td.text))

        logging.debug('Before reFormat, Shape of Histroy: %s',
                      numpy.shape(history))

        history = numpy.reshape(history, (-1, 21))

        logging.debug('After reFormat, Shape of Histroy: %s',
                      numpy.shape(history))

        logging.debug('Remove Header')
        history = numpy.delete(history, (0), axis=0)
        logging.debug(history)
        return history


history = numpy.zeros(shape=(0, 21))
matchCSV = pandas.read_csv('matchDate.csv', sep=',')
matchCSV = pandas.DataFrame(matchCSV)

try:
    # initial webCrawling
    webCrawling = WebCrawling()

    for index, row in matchCSV.iterrows():

        for matchIndex in range(row['RaceNo']):
            # Logic for debug use,
            if matchIndex > 2:
                break
            
            day = '{:0>4d}'.format(row['RaceDate1']) + '/' + '{:0>2d}'.format(
                row['RaceDate2']) + '/' + '{:0>2d}'.format(row['RaceDate3'])
            racecource = row['Racecourse']
            raceno = str(matchIndex + 1)
            # logging.info('day = %s, racecource=%s, raceno = %s ',
            #  day, racecource, raceno)
            temp_history = webCrawling.getHistory(day, racecource, raceno)
            logging.info(temp_history)
            history = numpy.concatenate((history, temp_history), axis=0)
        else:
            continue
        break


except Exception as ex:
    logging.error(ex)
finally:
    webCrawling.close()
    numpy.savetxt('history.csv', history, delimiter=',', fmt='%s')
