import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset   # from Hugging Face
from transformers import GPT2Tokenizer

ds = load_dataset("roneneldan/TinyStories") # ds: dictionary with train and validation split
print(ds)
print(ds['train'][0])


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token   # tells GPT-2 to use end-of-sequence token as padding token (filler to make all batches the same size)

# test it
sample = ds['train'][0]['text']
tokens = tokenizer.encode(sample)
print(f"\nStory length: {len(sample)} chars, {len(tokens)} tokens")

class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, block_size = 256, cache_path = 'tokens_cache.pt'):  # tiny stories data, tokenizer, sequence length, path for tokenized data
        self.block_size = block_size

        #  ensures that tokenized data is not redownloaded
        if os.path.exists(cache_path):
            self.tokens = torch.load(cache_path)
            print(f"Loaded tokens from cache: {len(self.tokens)}")
        else:
            # concatenate all stories into one long token stream
            print("Tokenizing dataset. . . ")

            def tokenize(example):
                return tokenizer(example['text'])

            tokenized = data.map(tokenize, batched = True, remove_columns = ['text'])   # tokenizes every story
            self.tokens = [id for seq in tokenized['input_ids'] for id in seq]  # flattens into a long list of token IDs
            torch.save(self.tokens, cache_path)
            print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) - self.block_size   # prevents each batch from running out of tokens

    # returns a pair of input (current tokens) and target (next tokens) sequences
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx: idx + self.block_size], dtype = torch.long)
        y = torch.tensor(self.tokens[idx + 1: idx + self.block_size + 1], dtype = torch.long)
        return x, y

train_dataset = TinyStoriesDataset(ds['train'], tokenizer, block_size = 256, cache_path = 'train_tokens.pt')
val_dataset = TinyStoriesDataset(ds['validation'], tokenizer, block_size = 256, cache_path = 'val_tokens.pt')

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False)