from twitterscraper import query_tweets
import datetime
import sys
import codecs,json


if __name__ == '__main__':

    #sys.stdout = open('/Users/anaghakaranam/Downloads/twitterscraper-master/log.txt', 'w')
    stock_query = query_tweets("HDFC or hdfc",10, begindate=datetime.date(2018,1,10),
                               enddate= datetime.date(2018,1,11),lang='en')
    file = open("hdfc_tweets.csv","w+")
    for tweet in stock_query:
        print(tweet.text)
        #file.write(str(tweet.text.encode('utf-8')))
        #file.write(tweet.encode('utf-8'))
    file.close()

    # with codecs.open('hdfc_tweets.csv','r','utf-8') as f:
    #     tweets = json.load(f, encoding = 'utf-8')
    #
    # print(tweets[5])