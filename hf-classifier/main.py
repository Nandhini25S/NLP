"""HF Classifier"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
from vocably.preprocessing.text import Preprocessor as pp

data = pd.read_csv('hf-classifier/data/movie_data.csv')
# print(data.head())

# Let's encode the target first
target_encoder = MultiLabelBinarizer()
target_encoder.fit(data['genre'])
y = target_encoder.transform(data['genre'])


#Now let's Preprocess the text
preprocessor = pp(remove_links = True)
data['overview'] = data['overview'].apply(preprocessor.normalize)

# Now let's tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',)
tokenized = data['overview'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(tokenized[0])
