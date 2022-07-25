import pandas as pd 
import numpy as np 
from tqdm import tqdm
from text import Preprocessing
import os


# Load the data
df = pd.read_json('/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/data/tweets.json')

X = df[df.columns.to_list()[0]]
y = df[df.columns.to_list()[1]]

preprocess = Preprocessing(lemmatize=True)

X = preprocess.tokenize(preprocess.Normalize(X))

y = y.replace(['Extremely Positive', 'Extremely Negative'], ['Positive', 'Negative'])
y = y.replace(['Positive', 'Negative','Neutral'], [2, 0,1])

classes = { 2: 'Positive',  0 : 'Negative', 1 : 'Neutral'}

del df

np.save("/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/data/X.npy",X)
print("X.npy saved")
np.save("/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/data/y.npy",y)
print("y.npy saved")