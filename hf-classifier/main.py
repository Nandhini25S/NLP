"""HF Classifier"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer

data = pd.read_csv('hf-classifier/data/movie_data.csv')
# print(data.head())

# Let's encode the target first
target_encoder = MultiLabelBinarizer()
target_encoder.fit(data['genre'])
y = target_encoder.transform(data['genre'])


#Now Tokenize the data using BERTT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoding = tokenizer.encode_plus(data['overview'].values[0], add_special_tokens = True,  
  truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")

print(len(encoding['input_ids'][0]))