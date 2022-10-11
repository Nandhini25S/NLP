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
Read blog [!here](https://towardsdatascience.com/what-is-an-encoder-decoder-model-86b3d57c5e1a#:~:text=Encoder%20decoder%20models%20allow%20for,This%20also%20works%20with%20videos.)
## Attention Mechanism
Attention mechanism is used to focus on the important parts of the input sequence.
In both encoder and decoder, the attention mechanism is used.
<img src = 'https://jalammar.github.io/images/t/Transformer_decoder.png'>
## 1) Embedding:

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

## 2) Self-Attention:

After getting a list of vectors

```Python
var = [[0.0522, 0.0178, 0.1694, ..., 0.0180, 0.0160, 0.0009],
       [0.1550, 0.1047, 0.0322, ..., 0.0555, 0.1045, 0.1085],
       [0.0610, 0.1158, 0.1463, ..., 0.1514, 0.0923, 0.0226],
       ...,
       [0.0114, 0.0556, 0.1169, ..., 0.1206, 0.0352, 0.0727],
       [0.0807, 0.0148, 0.1736, ..., 0.0337, 0.1697, 0.0101],
       [0.0308, 0.1114, 0.0200, ..., 0.1536, 0.1542, 0.1945]]

list[vectors] = [vector1, vector2, vector3, vector4, vector5]

for example: vector1 = [0.0522, 0.0178, 0.1694, ..., 0.0180, 0.0160, 0.0009]
Dimension = 512
5 * 512
List
elements = 2560
```
now role of Self attention,
for eg,
Consider a sentence, 

    English : cat is on the mat,it fall asleep.

what does the word 'it' refer to ? Human can say that it refers to cat that was mat
but when it comes to Algorithm, it is difficult to understand the context of the word 'it' in the sentence.
So we use self attention to understand the context of the word 'it' in the sentence.

math behind [!here](https://towardsdatascience.com/transformers-141e32e69591) and 
[!here too](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)

yt video [!here](https://peltarion.com/blog/data-science/self-attention-video)
understanding through [!blog](https://peltarion.com/blog/data-science/self-attention-video)
## 3) Multi Headed Self-Attention:

Multi headed self attention is uses Multiple Key, Value and Query vectors to calculate the attention.
for giving more attention to the words that are closer to each other.

## 4) Layer Normalization:
Layer normalization is used to normalize the output of the multi headed self attention layer.
## 5) Feed Forward Network:
Feed forward network is used to calculate the output of the encoder and decoder.
## 6) Encoder and Decoder Stacks:
Encoder and Decoder Stacks are used to increase the depth of the model.
## 7) Masked Self Attention:
Masked self attention is used to mask the attention of the padding tokens.
## 8) Cross Attention:
Cross attention is used to calculate the attention between the encoder and decoder.
## 9) Positional Encoding:
Positional encoding is used to encode the position of the word in the sentence.
## 10) Softmax:
Softmax is used to calculate the probability of the word in the sentence.
## 11) Loss Function:
Loss function is used to calculate the loss of the model. here we use cross entropy loss.
## 12) Optimizer:
Optimizer is used to optimize the loss function. here we use Adam optimizer.

----
## BERT (Bidirectional Encoder Representations from Transformers)
BERT is a technique for NLP pre-training developed by Google. 
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. 
BERT is a bidirectional transformer-based model which means it can be used for both NLP tasks like text classification,
question answering, and text generation. BERT is trained on a large corpus of unlabeled text and can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, 
such as question answering and language inference, without substantial task-specific architecture modifications.

Popular BERT models are 
    
        BERT-Base, uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
        BERT-Large, uncased: 24-layer, 1024-hidden, 16-heads, 340M parameters
        BERT-Base, cased: 12-layer, 768-hidden, 12-heads , 110M parameters
        BERT-Large, cased: 24-layer, 1024-hidden, 16-heads, 340M parameters
        BERT-Base, multilingual cased (New, recommended): 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
        BERT-Base, Chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
        BERT-Distilled, cased: 6-layer, 384-hidden, 4-heads, 33M parameters
        BERT-Distilled, uncased: 6-layer, 384-hidden, 4-heads, 33M parameters
        RoBERTa, cased: 12-layer, 768-hidden, 12-heads, 125M parameters
        RoBERTa, uncased: 12-layer, 768-hidden, 12-heads, 125M parameters
        RoBERTa, large, cased: 24-layer, 1024-hidden, 16-heads, 355M parameters
        RoBERTa, large, uncased: 24-layer, 1024-hidden, 16-heads, 355M parameters
        RoBERTa, base, cased: 12-layer, 768-hidden, 12-heads, 125M parameters
        RoBERTa, base, uncased: 12-layer, 768-hidden, 12-heads, 125M parameters
        RoBERTa, small, cased: 6-layer, 384-hidden, 6-heads, 17M parameters
        RoBERTa, small, uncased: 6-layer, 384-hidden, 6-heads, 17M parameters
    
BERT is a deep learning model that uses a technique called bidirectional encoding to read the text input.

