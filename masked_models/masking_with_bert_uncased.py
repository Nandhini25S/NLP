# Masking

from transformers import pipeline 
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
my_model = GPT2LMHeadModel.from_pretrained('gpt2')
my_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prediction  = pipeline('fill-mask', model = my_model)('Nandini is  [MASK]')
print(prediction)