import torch
from gpt2 import GPT, device, block_size
from transformers import GPT2Tokenizer

# load trained model
model = GPT().to(device)
model.load_state_dict(torch.load('model_final.pt', map_location = device))
model.eval()    # Note: used for validation and generation

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')   # uses GPT-2 model from Hugging Face; uses byte-pair encoding

prompt = "Once upon a time"
tokens = tokenizer.encode(prompt, return_tensors = 'pt').to(device)

with torch.no_grad():
    for _ in range(200):    # generate 200 new tokens (where each token is 3-4 characters on average for BPE)
        tokens_cropped = tokens[:, -block_size:]    # always only feed the last block_size tokens to prevent the model from running out of memory
        logits, _ = model(tokens_cropped)
        next_token = logits[:, -1, :].argmax(dim = -1, keepdim = True)  # (B, T, C) --> (B, 1). ":" keeps all batches, "-1" takes the last time step, and ":" keeps all token probabilities for that time step
        tokens = torch.cat([tokens, next_token], dim = 1)

print(tokenizer.decode(tokens[0].tolist()))