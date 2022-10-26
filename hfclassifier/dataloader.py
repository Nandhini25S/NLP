import torch
class CustomDataLoader(torch.utils.data.Dataset):

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def __init__(self, reviews, targets, tokenizer, max_len, device : str = 'cpu'):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        data = {key: torch.tensor(val).to(self.device) for key, val in self.reviews[item].items()}
        data['labels'] = torch.tensor(self.targets[item]).to(self.device)

        return data



class Tokenizer(object):
    def __init__(self):
        pass
        
    def build_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_tokens(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=512)

    