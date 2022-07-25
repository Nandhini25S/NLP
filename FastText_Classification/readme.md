# Fasttext Embdedding
 
 Fasttext was created by facebook research groups 
and is used to create embeddings for words.
    
    []: # Language: python
    []: # Path: FastText_Classification/Modules/gensim_vectorizers.py


## How is FastText different from gensim Word Vector ?

FastText differs in the sense that word vectors a.k.a word2vec treats every single word as the smallest unit whose vector representation is to be found but FastText assumes a word to be formed by a n-grams of character, for example, sunny is composed of [sun, sunn,sunny],[sunny,unny,nny]  etc, where n could range from 1 to the length of the word. This new representation of word by fastText provides the following benefits over word2vec or glove.

## Implementation for Pre-Trained wikipedia 

step 1: Download the pre-trained model from [here](https://fasttext.cc/docs/en/pretrained-vectors.html)
 select your preferred language
 for english [english.vec.zip](https://fasttext.cc/docs/en/english-vectors.html)
    
step 2: Download the model file from the link provided in the above link  

step 3: Extract the model file from the zip file.

step 4: Load the model file into the fasttext module.
 by creating class FastTextVectorizer
    
    []: # Language: python
    []: # Path: FastText_Classification/Modules/gensim_vectorizers.py

create object to that class

```python
from gensim_vectorizers import FastTextVectorizer
ft_vectorizer = FastTextVectorizer(model_path='/home/user/fasttext_model/wiki.en.bin')
```
    
