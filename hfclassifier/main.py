"""HF Classifier"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from vocably.preprocessing.text import Preprocessor as pp
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, AutoModelForSequenceClassification
from dataloader import CustomDataLoader, Tokenizer
from sklearn.model_selection import train_test_split
from rich.progress import track
torch.set_grad_enabled(True)
from rich import print as rprint
# torch.multiprocessing.set_start_method('spawn')
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cuda'
NUM_WORKERS = 4 if device == 'cpu' else 0
rprint(f'Using device ..{device}')




# load imdb dataset
data = pd.read_csv('hfclassifier/data/IMDB Dataset.csv', nrows=20)
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
    max_len = 512,
    device = device
)

testdataloader = CustomDataLoader(
    reviews= Xtest.to_numpy(),
    targets = ytest,
    tokenizer = tokenizer,
    max_len = 512,
    device = device
)


# Create the DataLoader for our training set.
train_data_loader = DataLoader(
    dataloader,
    batch_size = 16,
    num_workers = NUM_WORKERS,
    shuffle = True,
)

test_data_loader = DataLoader(
    testdataloader,
    batch_size = 16,
    num_workers = NUM_WORKERS
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

epochs = 5
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
    y_true.extend(batch['labels'].tolist())


torch.save(model, 'hfclassifier/model/model.pt')