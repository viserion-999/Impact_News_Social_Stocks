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
        sent[i] = analyser.polarity_scores(str(row['data']))
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
    dataframe['positive'] = dataframe['positive'].groupby(dataframe['date']).transform('mean')
    dataframe['negative'] = dataframe['negative'].groupby(dataframe['date']).transform('mean')
    dataframe['neutral'] = dataframe['neutral'].groupby(dataframe['date']).transform('mean')
    dataframe['compound'] = dataframe['compound'].groupby(dataframe['date']).transform('mean')
    dataframe = dataframe.drop_duplicates(subset = ['date'],keep= 'last')

    #lets sort the dataframe by date.
    dataframe.loc[:,'date'] = pd.to_datetime(dataframe.loc[:,'date'])
    #print(dataframe.info())
    #dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe.sort_values(by=['date'], inplace=True, ascending=True)

    return dataframe





#Read the 3 dataframes
'''
    Read the data into three seperate dataframes 
    economictimes, ndtv, thehindu.
   
'''

df_p1 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/newspaper_data/relianceindustries/results_econtimesArchive_reliance industries_1_1_2018_economictimes.indiatimes_content.csv")
df_p2 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/newspaper_data/relianceindustries/results_ndtvArchive_reliance industries_1_1_2018_ndtv.com_content.csv")
df_p3 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/newspaper_data/relianceindustries/results_thehinduArchive_reliance industries_1_1_2018_thehindu.com_content.csv")


#Bring dataframes to the form, date|Name of newspaper|Text|url


df_p1['Source'] = 'economic_times'
df_p2['Source'] = 'ndtv.com'
df_p3['Source'] = 'the_hi   ndu'

#in Step 2, drop the url.
df_p1 = df_p1.drop(['url','Source'], axis = 1)
df_p2 = df_p2.drop(['url','Source'], axis = 1)
df_p3 = df_p3.drop(['url','Source'], axis = 1)

print("number of rows in Economic Times",len(df_p1.axes[0]))
print("number of rows in NDTV",len(df_p2.axes[0]))
print("number of rows in Hindu",len(df_p3.axes[0]))
#merge the 3 data frames.
df_nwsp = pd.concat([df_p1,df_p2,df_p3], join= 'outer')


#print("number of rows after merging",len(df_nwsp.axes[0]))
#print("number of missing values",df_nwsp[pd.isna(df_nwsp).any(axis=1)])

#there are 4 rows with missing values, which we are dropping.
df_nwsp = df_nwsp.dropna(how = 'any')
#print("number of missing values after dropping",df_nwsp[pd.isna(df_nwsp).any(axis=1)])

'''
    Please note that although we havent written in the code,
    we have verified that there are no null values in any rows for
    individual dataframes & the merged datafame.

'''

# We have to convert the timestamp object to date & keep only the date part of it.
#print("Sample date is:",df_twt['timestamp'].head())
df_nwsp['date'] = pd.to_datetime(df_nwsp['date'])
df_nwsp['date'] = pd.to_datetime([d.date() for d in df_nwsp['date']])

#We keep the date format to dd-mm-YYYY

df_nwsp['date'] = df_nwsp['date'].dt.strftime('%d-%m-%Y')

#Apply Vader on this dataframe to find scores +ve, -ve, neutral and add to the df.
df_nwsp = analyze(df_nwsp)
df_nwsp = df_nwsp.drop(['data'], axis= 1)

df_nwsp = calculate_avg(df_nwsp)

#Here we round the values in all columns to 2 decimal places
df_nwsp['positive'] = df_nwsp['positive'].round(2)
df_nwsp['negative'] = df_nwsp['negative'].round(2)
df_nwsp['neutral'] = df_nwsp['neutral'].round(2)
df_nwsp['compound'] = df_nwsp['compound'].round(2)

# Set the index of the dataframe
df_nwsp.set_index('date', inplace= True)
#Now we have a dataframe with dates & scores.
print("Number of dates with values:",len(df_nwsp.axes[0]))   #139 :S
#We will add the missing dates & fill them with scores by looking back 15days.
idx = pd.date_range('01-01-2018', '01-01-2019')
df_nwsp.index = pd.DatetimeIndex(df_nwsp.index)
df_nwsp = df_nwsp.reindex(idx, fill_value= np.NaN)
print("finally:",df_nwsp.info())
# SHOULD WE LOOK BACK FOR NEWSPAPERS??
#print("Before interpolation number of missing values are: \n ", df_twt.isna().sum())
#WHICH INTERPOLATION?
#WHICH INTERPOLATION?
#WHICH INTERPOLATION?

#df_nwsp = df_nwsp.interpolate()
#print("After filling missing days: \n ",df_twt.isna().sum())

#print("After filling missing days: \n ",df_twt.isna().sum())
export_csv = df_nwsp.to_csv ("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Results/relianceindustries/ri_nws_result.csv", header=True)




