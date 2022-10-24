"""HF Classifier"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from vocably.preprocessing.text import Preprocessor as pp
# from hfclassifier.dataloader import CustomDataLoader as DL
# from hfclassifier.dataloader import Tokenizer as TK
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from rich.progress import track
torch.set_grad_enabled(True)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"
print(device)

# Tokenize input
class CustomDataLoader(torch.utils.data.Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        data = {key: torch.tensor(val).to(device) for key, val in self.reviews[item].items()}
        data['labels'] = torch.tensor(self.targets[item]).to(device)

        return data


class Tokenizer(object):
    def __init__(self):
        pass
        
    def build_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_tokens(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=512)


# load imdb dataset
data = pd.read_csv('hfclassifier/data/IMDB Dataset.csv', nrows=1000)
# print(data['sentiment'].value_counts())

# Let's encode the target first
target_encoder = LabelEncoder()
target_encoder.fit(data['sentiment'])
y = target_encoder.transform(data['sentiment'])
print('Target classes: ', target_encoder.classes_)


#Now let's Preprocess the text
preprocessor = pp(remove_links = True)
for idx in track(range(len(data['review'])), description="Preprocessing...", total=len(data['review'])):
    data['review'][idx] = preprocessor.normalize(data['review'][idx])


# Tokenize input
tokenizer = Tokenizer()
tokenizer.build_tokenizer(tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'))
data['review'] = data['review'].apply(tokenizer.get_tokens)
print('Tokenized Data ')

Xtrain , Xtest, ytrain, ytest = train_test_split(data['review'], y, test_size = 0.2, random_state = 42)

# Create the dataset and dataloader
dataloader = CustomDataLoader(
    reviews = Xtrain.to_numpy(),
    targets = ytrain,
    tokenizer = tokenizer,
    max_len = 512
)

testdataloader = CustomDataLoader(
    reviews= Xtest.to_numpy(),
    targets = ytest,
    tokenizer = tokenizer,
    max_len = 512
)


# Create the DataLoader for our training set.
train_data_loader = DataLoader(
    dataloader,
    batch_size = 4,
    num_workers = 4,
    shuffle = True,
)

test_data_loader = DataLoader(
    testdataloader,
    batch_size = 4,
    num_workers = 4
)

# Load pre-trained model (weights)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 2)
model.to(device)
# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Import nn modules
import torch.nn as nn
import torch.nn.functional as F

# Define the model
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 10
for epoch in range(epochs):
    for batch in track(train_data_loader, description = 'Training...', total = len(train_data_loader)):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs[0]
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} Loss {loss.item()}')

# Evaluate the model
model.eval()
y_pred = []
y_true = []
for batch in test_data_loader:
    outputs = model(**batch)
    y_pred.extend(torch.argmax(outputs.logits, dim = 1).tolist())
    y_true.extend(batch['targets'].tolist())

        




