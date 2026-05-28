import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import GPT2Model
from transformers import GPT2Tokenizer

# hyperparameters
vocab_size = 50257  # GPT-2 uses BPE tokenizer with a vocab size of 50257
block_size = 1024   # maximum context length
n_embd = 768        # embedding dimension
n_head = 12         # number of attention heads
n_layer = 12        # number of transformer blocks ('depth')
dropout = 0.0       # dropout probability (0.0% since we are doing inference, not training)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CausalSelfAttention(nn.Module):   # causal: can only attend to past (left) tokens

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0 # ensures all heads have equal dimensions

        # fused Q, K, V projection — matches HF's c_attn (768, 2304)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection — matches HF's c_proj (768, 768)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # causal mask; buffer is not trainable

    def forward(self, x):
        B, T, C = x.shape   # batch size, sequence length (number of tokens), embedding dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)    # splits c_attn from 2304 to 768 dimensions for Q, K, V each

        # reshape into heads so they can do their independent attention calculations
        head_size = C // self.n_head    # 768 / 12 = 64
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # formula can be found in "Attention is All You Need" paper
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)  # all future weights are no longer relevant
        out = wei @ v  # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, n_head, T, head_size) (B, T, n_head, head_size) -> (B, T, C)
        out = self.c_proj(out)
        return out

# feedforward network that adds nonlinearity
class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd) # expands space so that the neurons can "think" (more space for richer computations)
        self.c_proj = nn.Linear(4 * n_embd, n_embd) # compresses it back

    def forward(self, x):
        x = self.c_fc(x)    # expansion
        x = F.gelu(x, approximate = 'tanh')       # nonlinearity; GELU is a smoother version of ReLU and is used in GPT-2 - without it, multiple linear layers would be redundant
        x = self.c_proj(x)  # compression
        return x

# a block consists of an attention layer and a MLP layer, also known as "communication followed by computation"
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        # LayerNorm normalizes a vector such that it has a mean of 0 and a standard deviation of 1. Keeps values in a reasonable range at each step
        # one LayerNorm for attention and one for MLP
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # residual connection that prevents vanishing gradients and allows the model to learn identity functions if needed
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd) # word token embedding; embedding table that returns vector given token index
        self.wpe = nn.Embedding(block_size, n_embd) # word position embedding; the model has no inherent sense of order -> we add positional encoding
        self.h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]) # n_layer hidden layers
        self.ln_f = nn.LayerNorm(n_embd)    # final LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)  # language model head; converts numerical representation to token probabilities
        self.lm_head.weight = self.wte.weight   # matches input to output embeddings

    def forward(self, idx, targets = None):
        B, T = idx.shape    # breaks apart idx.shape[0] and idx.shape[1] into B and T , respectively
        tok_emb = self.wte(idx) # (B, T) --> (B, T, C)
        pos_emb = self.wpe(torch.arange(T, device = idx.device)) # (T, C) no need for batch since each position is the same across batches
        x = tok_emb + pos_emb   # input representation = token embedding + position embedding

        # runs the blocks n_layer times
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)    # final LayerNorm
        logits = self.lm_head(x)    #(B, T, C) --> (B, T, vocab_size)

        # used in training only; cross-entropy loss between predicted token probabilities and actual tokens
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

model = GPT()
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# load pre-trained Hugging Face weights into our model
def load_hf_weights(model):
    hf = GPT2Model.from_pretrained('gpt2')
    hf_sd = hf.state_dict()
    my_sd = model.state_dict()

    transposed = ['c_attn.weight', 'c_proj.weight', 'c_fc.weight']

    for key in my_sd.keys():
        if key == 'lm_head.weight':
            continue
        if key.endswith('tril'):
            continue
        if any(key.endswith(t) for t in transposed):
            my_sd[key].copy_(hf_sd[key].T)
        else:
            my_sd[key].copy_(hf_sd[key])

    print("weights loaded successfully")

load_hf_weights(model)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "The meaning of life is"
tokens = tokenizer.encode(prompt, return_tensors = 'pt').to(device) # returns a tensor that prioritizes moving into the GPU

model.eval()    # sets the model to evaluation, turning off dropout and training-only behaviors -- also no need to track gradients
with torch.no_grad():
    for _ in range(50): # generates 50 new tokens
        logits, _ = model(tokens)   # forward pass to get logits for next token. The _ ignores the loss
        next_token = logits[:, -1, :].argmax(dim = -1, keepdim = True)  # returns the token with the highest probability
        tokens = torch.cat([tokens, next_token], dim = 1)   # concatenates the new token to the existing tokens for the next iteration

print(tokenizer.decode(tokens[0].tolist()))