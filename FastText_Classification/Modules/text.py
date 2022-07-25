import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk import tokenize
from nltk.corpus import stopwords
#stemming
from nltk.stem import PorterStemmer
#lemmatization
from nltk.stem import WordNetLemmatizer
import re
import os

#logger 
import logging
logging.basicConfig(
    filename=os.path.join("logs", ''), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


class Preprocessing:
    def __init__(self , lemmatize = False):
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        

    def text_normalize(self, text):
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        # text re to select only alphabets
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        # text to replace emails and urls
        # text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'\S*\s?(http|https)\S*', '', text)
        # text to replace emojis
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    #remove punctuation
    def Normalize(self, data):
        logging.info("text Normalise")
        return data.apply(self.text_normalize)
    

    #tokenize and remove stopwords and lemmatize or stem
    def text_tokenize(self, text):
        if self.lemmatize == False:
            return tokenize.word_tokenize(self.text_stem(self.text_remove_stopwords(text)))
        else:
            return tokenize.word_tokenize(self.text_lemmatize(self.text_remove_stopwords(text)))

    def tokenize(self, data):
        logging.info("text to tokenize")
        return data.apply(self.text_tokenize)
    
    def text_remove_stopwords(self, text):
        #stopwords without not is'nt
        whitelist = ["not"]
        text = [word for word in text.split() if word not in stopwords.words('english') or word in whitelist]
        return ' '.join(text)
    
    #stemming
    def text_stem(self, text):
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in text]
    
    #lemmatization
    def text_lemmatize(self, text):
        text = [self.lemmatizer.lemmatize(word,pos = 'v') for word in text.split()]
        return ' '.join(text)


