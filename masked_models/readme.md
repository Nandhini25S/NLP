### Masked Language Models (NLP)

MLM (Masked Language Models) consists of giving BERT a sentence and optimizing the weights inside BERT to output the same sentence on the other side. This is done by masking the words in the sentence. The masking is done by replacing the word with a token (MASK- token). The token is a special token that is not part of the vocabulary. 

![alt](https://miro.medium.com/max/770/1*phTLnQ8itb3ZX5_h9BWjWw.png)

### Working Steps

step 1: import necessary libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
``` 

step 2: create tokenizer and model

Here we are going to use the `tokenizer` from the `transformers` library.
`tokenizer` is a class that is used to tokenize the sentences.

```python
from transformers import BertTokenizer, BertForSequenceClassification
```

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```
tokenizer recieves the pretrained model and returns a tokenizer object.
in three different tensors, the first tensor is the word ids, the second tensor is the attention mask, and the third tensor is the token type ids.

what is input_ids, attention_mask and token_types ? 

-> input_ids are mapping tokens to the respective words
-> input_id of a sentence starts with [ 'CLS' ] token with value 101.
-> input_id of a senetcne ends with [ 'SEP' ] tokens with value 102.


-> attention mask provides attention to the tokens 
-> token_types for two sequence for example 


    [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]

python code 

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "My job is to eliminate stopwords "
sequence_b = "removing stop words is my job"

encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])
```

[ CLS ] My job is to eliminate stopwords [ SEP ] removing stop words is my job [ SEP ]

step 3: Create Masks using [ MASK ] tokens
    
```python
    from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
    my_model = GPT2LMHeadModel.from_pretrained('gpt2')
    my_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "Hello world"
    encoded_dict = my_tokenizer(text)

```


```python
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead
prediction = pipeline("fill-mask", model=my_model, tokenizer=my_tokenizer, prompt=text)
```
or 

```python 
from transformers import pipeline 
prediction  = pipeline('fill-mask', model = 'bert-uncased')('Hello [MASK]')
```

