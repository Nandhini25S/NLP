
import json
import pandas as pd
import numpy as np
from pprint import pprint

# read dataframe to json
# def to_json(df):
#     return df.to_json(orient='records') 


data = pd.read_csv('/media/senju/1.0 TB Hard Disk/Text_Classification_NLP/data/Corona_NLP_train.csv',encoding='latin-1')
print(data[['OriginalTweet','Sentiment']].head())
# json_df = data[['OriginalTweet','Sentiment']].to_json(orient='records')
# pprint(json_df)
data[['OriginalTweet','Sentiment']].to_json(r'data/tweets.json',orient='records')
