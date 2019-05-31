import pandas as pd
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import warnings

warnings.filterwarnings("ignore")
analyser = SentimentIntensityAnalyzer()

def analyze(dataframe):

    sent = {}
    pos = []
    neg = []
    com = []
    neu = []
    #print(len(dataframe))
    i  = 0
    for index, row in dataframe.iterrows():
        i+=1
        sent[i] = analyser.polarity_scores(str(row['text']))
        pos.append(sent[i]['pos'])
        neg.append(sent[i]['neg'])
        neu.append(sent[i]['neu'])
        com.append(sent[i]['compound'])

    data_tuples = list(zip(pos, neg,neu, com))
    senti_df = pd.DataFrame(data_tuples, columns= ['pos','neg','neu','com'])
    dataframe.index.names = ['index']
    senti_df.index.names = ['index']
    #print("length of df senti",len(senti_df))

    #print("Index is:", dataframe.index.name)
    #pd.concat([dataframe, senti_df], axis=1)
    dataframe['positive'] = pd.Series(senti_df['pos'])
    dataframe['negative'] = pd.Series(senti_df['neg'])
    dataframe['neutral'] = pd.Series(senti_df['neu'])
    dataframe['compound'] = pd.Series(senti_df['com'])
    #dataframe = dataframe.join(senti_df, how = 'outer')
    print("length of dataframe is:", len(dataframe))
    print("Has nan values?", dataframe.isnull().values.any())

    return dataframe

def calculate_avg(dataframe):
    dataframe['positive'] = dataframe['positive'].groupby(dataframe['Date']).transform('mean')
    dataframe['negative'] = dataframe['negative'].groupby(dataframe['Date']).transform('mean')
    dataframe['neutral'] = dataframe['neutral'].groupby(dataframe['Date']).transform('mean')
    dataframe['compound'] = dataframe['compound'].groupby(dataframe['Date']).transform('mean')
    dataframe = dataframe.drop_duplicates(subset = ['Date'],keep= 'last')

    #lets sort the dataframe by date.
    dataframe.loc[:,'Date'] = pd.to_datetime(dataframe.loc[:,'Date'])
    #print("unique rows after avging",dataframe.info())
    #dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    dataframe.sort_values(by=['Date'], inplace=True, ascending=True)
    print("inside func",dataframe.info())
    return dataframe


#Read the 3 dataframes
'''
    Read the data into three seperate dataframes & drop columns not required.
    We only keep the columns: timestamp, text, likes, replies & retweets.
'''
#print("Here we can setup input of stock to perform twitter analysis!")
df_p1 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/twitter_data/relianceindustries/reliance_tweets_q1.csv")
df_p2 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/twitter_data/relianceindustries/reliance_tweets_q2.csv")
df_p3 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/twitter_data/relianceindustries/reliance_tweets_q3.csv")

#dropping unnecessary columns
#For now we are dropping likes, retweets and replies because they are biased
df_p1 = df_p1.drop(['user','fullname','tweet-id','url','html','likes','replies','retweets'], axis = 1)
df_p2 = df_p2.drop(['user','fullname','tweet-id','url','html','likes','replies','retweets'], axis = 1)
df_p3 = df_p3.drop(['user','fullname','tweet-id','url','html','likes','replies','retweets'], axis = 1)

#print("desc:",df_p1.info())
#print("desc:",df_p2.info())
#print("desc:",df_p3.info())

#merge the 3 data frames.
df_twt = pd.concat([df_p1,df_p2,df_p3], join= 'outer')

'''
    Please note that although we havent written in the code,
    we have verified that there are no null values in any rows for
    individual dataframes & the merged datafame.

'''


# We have to convert the timestamp object to date & keep only the date part of it.
#print("Sample date is:",df_twt['timestamp'].head())
#Sample is: 2018-03-30 18:19:48

df_twt['Date'] = pd.to_datetime(df_twt['timestamp'])
df_twt = df_twt.drop(['timestamp'], axis= 1)
df_twt['Date'] = pd.to_datetime([d.date() for d in df_twt['Date']])
#We keep the date format to dd-mm-YYYY

df_twt['Date'] = df_twt['Date'].dt.strftime('%d-%m-%Y')

## Till here it is data preprocessing.
#print("Info about final dataframe before cal",df_twt.info())
#Apply Vader on this dataframe to find scores +ve, -ve, neutral and add to the df.
df_twt = analyze(df_twt)
#Lets drop the text also now!
df_twt = df_twt.drop(['text'], axis= 1)

df_twt = calculate_avg(df_twt)

#Here we round the values in all columns to 2 decimal places
df_twt['positive'] = df_twt['positive'].round(2)
df_twt['negative'] = df_twt['negative'].round(2)
df_twt['neutral'] = df_twt['neutral'].round(2)
df_twt['compound'] = df_twt['compound'].round(2)
print("Final dataframe:",df_twt.info())
# Set the index of the dataframe
df_twt.set_index('Date', inplace= True)
#Now we have a dataframe with dates & scores.

#We will add the missing dates & fill them with scores by looking back 15days.
idx = pd.date_range('01-01-2018', '01-01-2019')
df_twt.index = pd.DatetimeIndex(df_twt.index)
df_twt = df_twt.reindex(idx, fill_value= np.NaN)
print("Final dataframe:",df_twt.info())

#print("Before interpolation number of missing values are: \n ", df_twt.isna().sum())
#WHICH INTERPOLATION?
#WHICH INTERPOLATION?
#WHICH INTERPOLATION?

#df_twt = df_twt.interpolate()
#print("After filling missing days: \n ",df_twt.isna().sum())
export_csv = df_twt.to_csv ("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Results/relianceindustries/ritw_result.csv", header=True)

print("Twitter Module is Done! 8-D")



