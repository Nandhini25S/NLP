import pandas as pd 
import numpy as np 
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import io
import logging
import os


logging.basicConfig(
    filename=os.path.join("fasttext_logs", ''), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

class Word2Vec_vectorize:
    def __init__(self, df, column, model_name, vector_size, window_size, min_count, workers, negative, iter):
        self.df = df
        self.column = column
        self.model_name = model_name
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers
        self.negative = negative
        self.iter = iter

    def train(self):
        model = Word2Vec(self.df[self.column], size=self.vector_size, window=self.window_size, min_count=self.min_count, workers=self.workers, negative=self.negative, iter=self.iter)
        model.save(self.model_name)
        logging.info("trained word to vector")
        return model


class FastText_vectorize:
    def __init__(self,file_path : str = None) -> None:
        self.file_path = file_path
        self.dimension = 300
        self.data = {}
        pass

    def _build_vector(self):
        f = io.open(self.file_path,'r',encoding = 'utf-8',newline='\n',errors = 'ignore')
        # n, d = map(int, f.readline().split())
        # print(d)
        for line in tqdm(f,colour = 'red'):
            tokens = line.strip().split(' ')
            self.data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        # with time log
        logging.info("built word to vector")
        # return data
    
    def _get_dimension(self):
        return self.dimension

    def _get_vector(self,word : str):
        if word in self.data and self.data != {}:
            return self.data[word]
        else:
            return np.zeros(self.dimension)
    
    def _get_vector_list(self,word_list : list):
        if self.data == {}:
            self._build_vector()
        if word_list == []:
            return np.zeros(self.dimension)
        return np.array([self._get_vector(word) for word in word_list])

    def padding_truncate(self,vector_list : list,max_length : int = 100):
        if len(vector_list) > max_length:
            return np.transpose(vector_list[:max_length])
        else:
            return np.concatenate((vector_list,np.zeros((max_length-len(vector_list),self.dimension)))).T
    
    #Thanks

    

    