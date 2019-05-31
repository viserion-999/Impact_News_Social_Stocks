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


# Read the 3 dataframes
'''
    Read the data into three seperate dataframes 
    economictimes, ndtv, thehindu.

'''

df_p1 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/googlesearch_data/relianceindustries/results_ businesstoday.in_reliance industries_01_01_2018_businesstoday.in_content.csv")
df_p2 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/googlesearch_data/relianceindustries/results_ in.reuters.com_reliance industries_01_01_2018_in.reuters.com_content.csv")
df_p3 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/googlesearch_data/relianceindustries/results_ livemint_reliance industries_01_01_2018_livemint.com_content.csv")
df_p4 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/googlesearch_data/relianceindustries/results_ moneycontrol.com_reliance industries_01_01_2018_moneycontrol.com_content.csv")
df_p5 = pd.read_csv("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Data/googlesearch_data/relianceindustries/results_ news18.com_reliance industries_01_01_2018_news18.com_content.csv")

#in Step 2, drop the url.
df_p1 = df_p1.drop(['url'], axis = 1)
df_p2 = df_p2.drop(['url'], axis = 1)
df_p3 = df_p3.drop(['url'], axis = 1)
df_p4 = df_p4.drop(['url'], axis = 1)
df_p5 = df_p5.drop(['url'], axis = 1)

print("number of rows - business today",len(df_p1.axes[0]))
print("number of rows - reuters",len(df_p2.axes[0]))
print("number of rows - livemint",len(df_p3.axes[0]))
print("number of rows - money control",len(df_p4.axes[0]))
print("number of rows - news 18",len(df_p5.axes[0]))

df_googs = pd.concat([df_p1,df_p2,df_p3,df_p4,df_p5], join= 'outer')


#print("number of rows after merging",len(df_googs.axes[0]))
#print("number of missing values",df_googs[pd.isna(df_googs).any(axis=1)])
df_googs = df_googs.dropna(how = 'any')

# We have to convert the timestamp object to date & keep only the date part of it.
#print("Sample date is:",df_twt['timestamp'].head())
df_googs['date'] = pd.to_datetime(df_googs['date'])
df_googs['date'] = pd.to_datetime([d.date() for d in df_googs['date']])

#We keep the date format to dd-mm-YYYY

df_googs['date'] = df_googs['date'].dt.strftime('%d-%m-%Y')

#Apply Vader on this dataframe to find scores +ve, -ve, neutral and add to the df.
df_googs = analyze(df_googs)
df_googs = df_googs.drop(['data'], axis= 1)

df_googs = calculate_avg(df_googs)

#Here we round the values in all columns to 2 decimal places
df_googs['positive'] = df_googs['positive'].round(2)
df_googs['negative'] = df_googs['negative'].round(2)
df_googs['neutral'] = df_googs['neutral'].round(2)
df_googs['compound'] = df_googs['compound'].round(2)

print("Number of dates with values:",len(df_googs.axes[0])) #183 :S
# Set the index of the dataframe
df_googs.set_index('date', inplace= True)
#Now we have a dataframe with dates & scores.
# We will add the missing dates & fill them with scores by looking back 15days.
idx = pd.date_range('01-01-2018', '01-01-2019')
df_googs.index = pd.DatetimeIndex(df_googs.index)
df_googs = df_googs.reindex(idx, fill_value=np.NaN)

# SHOULD WE LOOK BACK FOR NEWSPAPERS??
# print("Before interpolation number of missing values are: \n ", df_twt.isna().sum())
# WHICH INTERPOLATION?
# WHICH INTERPOLATION?
# WHICH INTERPOLATION?
print("finally:",df_googs.info())
#df_googs = df_googs.interpolate()
# print("After filling missing days: \n ",df_twt.isna().sum())

export_csv = df_googs.to_csv ("/Volumes/ANAGHA/Stock_prediction/Modules/Sentiment_analysis_Module1and2/Results/relianceindustries/ri_gs_result.csv", header=True)

