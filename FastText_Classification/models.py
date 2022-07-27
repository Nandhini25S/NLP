"""
Notebook to python file
"""
import numpy as np
import pandas as pd
from Modules.gensim_vectorizers import FastText_vectorize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , Bidirectional , Dropout
#to_categorical
from tensorflow.keras.utils import to_categorical

X = np.load('/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/data/X.npy',allow_pickle= True)
y = np.load('/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/data/y.npy',allow_pickle= True)

vec = FastText_vectorize("/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/fasttext/wiki-news-300d-1M.vec")
vec._build_vector()
X, y = X[:3500] , y[:3500]
vec_list = np.array([vec.padding_truncate(vec._get_vector_list(X[i]),max_length=20) for i in range(X.shape[0])])
vec_list = np.array(vec_list)
(SequenceLegnth , EmbeddingLength) = vec_list[0].shape

print(f"SequenceLegnth = {SequenceLegnth}, EmbeddingLength = {EmbeddingLength}")
np.save('/media/senju/1.0 TB Hard Disk/NLP/FastText_Classification/data/X_vec.npy',vec_list)

y = to_categorical(y)

def build_model(optim = "adam", SequenceLegnth = 100, EmbeddingLength = 300,output_shape = 2):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(SequenceLegnth, EmbeddingLength)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(units=output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

# model.summary()
model = build_model(optim = "adam", SequenceLegnth = SequenceLegnth, EmbeddingLength = EmbeddingLength,output_shape = y.shape[1])
model.fit(vec_list, y, epochs=10, batch_size=32,validation_split=0.2)

model.save('/media/senju/1.0 TB Hard Disk/NLP/models/Tweet_Positive_Negative.h5')

