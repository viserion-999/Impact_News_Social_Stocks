import pandas as pd

df = pd.read_csv("reliance_tweets_q3.csv",delimiter = ',' )

#print(df.info())
print(df['text'][1000])
