# nanoGPT Walkthrough

**(1) Abstract**

*nanoGPT* is a tiny version of a generative pre-trained transformer (GPT), with its use of causal self-attention, multi-head attention, and an architecture that closely resembles GPT-2.  
The structure of *nanoGPT* can be thoroughly explained by the paper "Attention is All You Need".  
Additionally, refer to the paper "Language Models are Unsupervised Multitask Learners".

In *nanoGPT*, there are two files (each ~300 lines) that define the model: `train.py`, which provides the training loop, and `model.py`, a GPT model definition.

Andrej Karpathy's goal was to create a clean and minimal GPT-2 reproduction.

**(2) Data Tokenization**

When we feed a model text, audio, or images, it needs a way to convert the human-interpretable data into numbers.  
We do so using a tokenizer.
The tokenizer looks up the word in a table, assigning each token with a token ID.
The token ID is then converted into a vector via an embedding table that becomes a vector representation.  

It is important to note that machine learning models have no inherent sense of order. For example, "a man bit a dog" contains the same tokens as "a dog bit a man". This is because attention is permutation-invariant, which means they cannot differentiate their order without additional assistance.
To resolve this, we introduce positional encoding, which gives each token a position in the sequence.

Our input representation is the embedded tokens plus the positional embedding. When we feed this data into our model, it is given a token vector that the machine knows what it means and where it is located in the sequence.

**(3) Dataloader**

The `DataLoader` sorts data into two categories: training and validation data. The model is trained on training data, improving its model accuracy over time. We then use evalation mode to analyze the model's cross-entropy loss as we test it on previously unseen data (validation data).

It returns a pair of inputs and targets, respectively labeled `x` and `y`.  
The input sequence is a list of the current tokens with a length of block size.  
The target sequence is the input sequence shifted to the right by one token.
This is done so the model know that, given a list of inputs, it should be able to predict the targets.

**(4) Model Architecture**

TBD