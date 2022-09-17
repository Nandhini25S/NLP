import json
# from posixpath import split
from tqdm import tqdm
from datasets import load_dataset

# convert to this format
'''TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]'''

def convert_to_spacy(dataset):
    Train_data = []
    for i in range(len(dataset['tokens'])):
        entities = []
        for j in tqdm(range(len(dataset['tokens'][i])),colour='green'):
            if dataset['ner_tags'][i][j] != 'O':
                entities.append((j, j+1, dataset['ner_tags'][i][j]))
        Train_data.append((' '.join(dataset['tokens'][i]), {'entities': entities}))
    return Train_data


dataset = load_dataset('wikiann','en',split='train')
print(' '.join(dataset['tokens'][0]),dataset['spans'][0])
print(convert_to_spacy(dataset[:3]))