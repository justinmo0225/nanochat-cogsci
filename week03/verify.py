import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gpt2 import GPT, load_hf_weights

# load your model
my_model = GPT()
load_hf_weights(my_model)
my_model.eval()

# load HF model
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_model.eval()

# same prompt with numerical tolerance
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode("The meaning of life is", return_tensors = 'pt')

# forward pass through both
with torch.no_grad():
    my_logits, _ = my_model(tokens)
    hf_logits = hf_model(tokens).logits

# compare
print("max difference:", (my_logits - hf_logits).abs().max().item())
print("match:", torch.allclose(my_logits, hf_logits, atol = 1e-4))

# VERIFY: should return 'match: True' with a very small max difference