#read json file to dataframe
import json
import pandas as pd
from requests import head

def from_json(json_file):
    return pd.read_json(json_file, orient='records')

#read json file
with open('data/tweets.json') as f:
    file = json.load(f)
    df = pd.DataFrame(file)

print(df.head())
