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

transformer provided lot of improvements over the RNNs and CNNs. The transformer architecture is based on the attention mechanism. The attention mechanism is used to focus on the important parts of the input sequence. The attention mechanism is used to calculate the context vector. The context vector is used to calculate the output of the transformer. The transformer architecture is used in many NLP tasks like text classification, text summarization, text generation, machine translation, etc. okay !! blah blah blah !! very confusing right !! !! consider the following example to understand the transformer architecture.

consider a sentence, 

    English : cat is on the mat 


to transformer the above sentence into another language(Machine Translation)

    Tamil : Pūṉai pāyil uḷḷatu

Transformer has 2 main blocks Encoder block 

    Encoder block : Encoder block is used to encode the input sequence. The encoder block is used to calculate the context vector. The context vector is used to calculate the output of the transformer.
    

    Decoder block : Decoder block is used to decode the context vector. The decoder block is used to calculate the output of the transformer.

Deep learning framework like PyTorch and TensorFlow is used to implement the transformation model. Even in Computer Vision Transformer architecture is used to outperform CNNs. We reproduce Transformer architecture with different deep learning frameworks and achieve higher accuracy on the classification task.
According to Jay Alammar's blog, transformer has six main components, i will explain each one of them below and i also referred to Narrated transformers video from Jay Alammar.

before that let's see encoder and decoder stacks
Encoder and Decoder stacks : 

    Encoder : Encoder receives input and calculate context vector
    Decoder : Decoder receives context vector and gives output based on that vector
    Encoder and Decoder are stacks of encoders and decoders with the same parameters

Examples of Decoder stakcs
    
    GPT model (dialog human chat dataset, web scraping data, language modeling)
    GPT-2 model (dialog human chat dataset)
    GPT3.ai model (dialog human chat dataset, web scraping data, language modeling)

GPT - 2 (openAI transformers) Huge Language Model which is trained  on  the web scraping data. with 36 blocks in each stack.

Examples of Encoder Stacks

    BERT Models ( classification, another NLP task)
    Bart Models ( chat bot trained on publicly available chat data like reddit and conversations on product hunt)
    Longformer ( BERT based models trained on long inputs ( not possible with the BERT))
    Huge BERT models ( preprocessed text by replacing randomly selected words with [MASK] token and evaluating against the original sentence ) Here we can interpret as [MASK] token as a wildcard.
    XLNet Models ( models trained with combined objective (self-supervised object), original uncombined objective (Standard BERT) & next sentence prediction task)
So we got clarity about encoder decoder, Now we reduce complexity of understanding the architecture by simplifying things a little by removing cross-entities and heads parts.
 ## 1) Self attention


## 2) Embedding

The encoder and the decoder contains the embedding layer for embedding the source language and target language respectively.

Token embedding (wte)
    
        Token embedding (wte) : Token embedding (wte) is used to embed the word in sequence when the input is a word.(Encoder and Decoder have two wte's).

Positional embedding(wpe)

        Positional embedding(wpe) : Positional embedding(Wpe) is used to account for position in the sentence. The positional code(PE) is combined with the word embedding to produce the input word embedding vector.

        The Shawshank Redemption
        Produces an input embedding like below
        The Shawshank Redemption
    
        [[0.0522, 0.0178, 0.1694, ..., 0.0180, 0.0160, 0.0009],
         [0.1550, 0.1047, 0.0322, ..., 0.0555, 0.1045, 0.1085],
         [0.0610, 0.1158, 0.1463, ..., 0.1514, 0.0923, 0.0226],
         ...,
         [0.0114, 0.0556, 0.1169, ..., 0.1206, 0.0352, 0.0727],
         [0.0807, 0.0148, 0.1736, ..., 0.0337, 0.1697, 0.0101],
         [0.0308, 0.1114, 0.0200, ..., 0.1536, 0.1542, 0.1945]]

Each Word (token) gets a unique embedding based on it position relative to other words in the sentence.
         

## 3) Multi Headed Self-Attention
