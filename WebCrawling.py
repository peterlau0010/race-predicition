from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.chrome.options import Options
import bs4
import logging
import time
import re
import numpy
import pandas


class WebCrawling:
    logging.basicConfig(filename='./Log/WebCrawling.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

    def __init__(self):
        self.browser = webdriver.Safari()

    def close(self):
        self.browser.quit()

    def request(self, url):
        self.browser.get(url)
        try:
            logging.info('URL with %s', url)
            page_source = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div/div[5]/table")))
            html_source = bs4.BeautifulSoup(self.browser.page_source, 'lxml')
            return html_source
        except TimeoutException as ex:
            logging.error('Loading took too much time!')
            logging.error('Exception found %s', ex)
            logging.error('URL with %s', url)
            return

        # time.sleep(10)

    def reFormatText(self, text):
        # Trim space, remove line break, remove double space
        return re.sub(' +', ' ', text.replace('\n', ' ').replace('\r', '').replace(',', '').replace('HK$ ', '').strip())

    def getHistory(self, day, racecourse, raceno):
        url_base = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?'
        url_raceDate = 'RaceDate=' + day
        url_raceCourse = '&Racecourse=' + racecourse
        url_raceNo = '&RaceNo=' + raceno
        url = url_base + url_raceDate + url_raceCourse + url_raceNo

        logging.debug('Access Url: %s', url)

        html_source = self.request(url)

        # Initital history list
        history = []
        history_header = []
        if html_source is None:
            history = numpy.zeros(shape=(0, 21))
            return history

        table = html_source.find(
            name='table', class_='f_tac table_bd draggable')

        matchDetail = html_source.find(name='tbody', class_='f_fs13')

        # Add Index value to history
        history_header.append(day)
        history_header.append(racecourse)
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
        try:
            history = numpy.reshape(history, (-1, 21))
        except Exception as ex:
            logging.error('Exception found %s', ex)
            history = numpy.zeros(shape=(0, 21))
            return history

        logging.debug('After reFormat, Shape of Histroy: %s',
                      numpy.shape(history))

        logging.debug('Remove Header')
        history = numpy.delete(history, (0), axis=0)
        logging.debug(history)
        return history


history = numpy.zeros(shape=(0, 21))
matchCSV = pandas.read_csv('./Raw Data/matchDate.csv', sep=',')
matchCSV = pandas.DataFrame(matchCSV)

try:
    # initial webCrawling
    logging.info('Initital Web Crawling - Launch Selenium Browser')
    webCrawling = WebCrawling()

    for index, row in matchCSV.iterrows():

        for matchIndex in range(row['RaceNo']):
            # Logic for debug use,
            # if matchIndex > 2:
            #     break

            day = '{:0>4d}'.format(row['RaceDate1']) + '/' + '{:0>2d}'.format(
                row['RaceDate2']) + '/' + '{:0>2d}'.format(row['RaceDate3'])
            racecourse = row['Racecourse']
            raceno = str(matchIndex + 1)

            logging.info('Access web with parameters, day = %s, racecourse=%s, raceno = %s ',
                         day, racecourse, raceno)

            temp_history = webCrawling.getHistory(day, racecourse, raceno)
            logging.info('Get history record with shpe: %s',
                         numpy.shape(temp_history))

            history = numpy.concatenate((history, temp_history), axis=0)
            logging.info('Added to history list')
        else:
            continue
        break


except Exception as ex:

    logging.error('Exception found %s', ex)
finally:

    logging.info('Close Selenium Browser')
    webCrawling.close()

    logging.info('Ready to write csv with histroy shape: %s',
                 numpy.shape(history))
    headers = ['date','raceCourse','raceNo','going','raceName','road','money','class','dist','plc','horseNo','horseName','jockey','trainer','awt','dhw','draw','lbw','runPos','finishTime','odds']
    headers = ','.join(headers)
    numpy.savetxt('./Raw Data/history.csv', history, delimiter=',', fmt='%s',header=headers, comments='')
    logging.info('Finished write csv')
