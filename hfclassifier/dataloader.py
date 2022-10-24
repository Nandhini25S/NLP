import torch


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
        data = {key: torch.tensor(val) for key, val in self.reviews[item].items()}
        data['labels'] = torch.tensor(self.targets[item])

        return data


class Tokenizer(object):
    def __init__(self):
        pass
        
    def build_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_tokens(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=512)

    