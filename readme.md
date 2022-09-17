# Natural Language Processing (Modules)
# #############################################################################################################################

## Install Dependencies 

create env conda or pip env
```bash
    conda create -n nlp_env python=3.8
    conda activate nlp_env
    pip install -r requirements.txt
```

pip env 
```bash
    python3 -m venv nlp_env
    source nlp_env/bin/activate
    pip install -r requirements.txt
```
To install packages in a virtual environment:

```bash
pip install -r requirements.txt
```
## 1) Text Normalization

### 1.1) Text Cleaning

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
```
## 1.2) Text cleaning using moduls

```python

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
```

```python
    def text_clean(self, text):
        text = self.text_normalize(text)
        text = self.text_remove_stopwords(text)
        text = self.text_tokenize(text)
        return text
```

## 2) Stop Word Removal

```python
   def text_remove_stopwords(self, text):
        #stopwords without not is'nt
        whitelist = ["not"]
        text = [word for word in text.split() if word not in stopwords.words('english') or word in whitelist]
        return ' '.join(text)
```


## 3) Tokenization and Lemmatization


```python

    def text_tokenize(self, text):
        # tokenize text
        tokens = word_tokenize(text)
        # lemmatize each word - if it is not a noun, verb, adjective, adverb
        lemmatizer = WordNetLemmatizer()
        # stem each word - if it is not a verb, adjective, adverb
        stemmer = PorterStemmer()
        # remove stopwords
        text = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in stopwords.words('english')]
        return text
```

Note you can use Stemming also


## 4) Vectorization

### 4.1) Count Vectorizer

```python
    from sklearn.feature_extraction.text import CountVectorizer
```

### 4.2) TF-IDF Vectorizer

```python
    from sklearn.feature_extraction.text import TfidfVectorizer
```

### 4.3) Word2vec

    
```python
    from gensim.models import Word2Vec
```

### 4.4) AutoTokenizers 

Bert (Attention based models)

## Attention is All you need : )

### Transformer Architecture

from packt publication and by refering two to three articles i have created this architecture

the attention is all you need is an article, written by [Google Brain](https://research.google/teams/brain/) and Google researchers [Ashish Vaswani](https://twitter.com/ashishvaswani), [Noam Shazeer](https://twitter.com/nshazeer), [Niki Parmar](https://twitter.com/nikisparmar), [Jakob Uszkoreit](https://twitter.com/jakobuszkoreit), [Llion Jones](https://twitter.com/llionjones), [Aidan N Gomez](https://twitter.com/aidangomez), [Lukasz Kaiser](https://twitter.com/lukaszkaiser), [Illia Polosukhin](https://twitter.com/illyap) in 2017. The paper is available on [arXiv](https://arxiv.org/abs/1706.03762) and [GitHub](
[!Attention is all you need](https://arxiv.org/abs/1706.03762)

this transformer outperformed other transformer architecture so the point here is how important the attention is in the transformer architecture. Transformer is the key component of NLP and it is used in many NLP tasks like text classification, text summarization, text generation, machine translation, etc.

## 1) Transformer Architecture
<!-- insert image -->
<img src = 'https://imgs.search.brave.com/9siRrM0u4OXw2mkcKMVLcVLgTQGGysSRy9-nd00Wtww/rs:fit:626:575:1/g:ce/aHR0cHM6Ly9taXJv/Lm1lZGl1bS5jb20v/bWF4LzEyNTIvMSpK/dUdaYVpjUnRtcnRD/RVBZOHFmc1V3LnBu/Zw'>



    