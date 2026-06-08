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

The `DataLoader` sorts data into two categories: training and validation data. The model is trained on training data, improving its model accuracy over time. We then use evaluation mode to analyze the model's cross-entropy loss as we test it on previously unseen data (validation data).

It returns a pair of inputs and targets, respectively labeled `x` and `y`.  
The input sequence is a list of the current tokens with a length of block size.  
The target sequence is the input sequence shifted to the right by one token.
This is done so the model knows that, given a list of inputs, it should be able to predict the targets.

**(4) Model Architecture**

After the input is tokenized and given positional embeddings, the data is then is put into causal self-attention, which evaluates the relevancy of each token in a sequence (causal means each token can only perceive relevancy in relation to itself and all past tokens, not future tokens). Using masking, all future tokens are set to negative infinity. That way, applying softmax results in all future tokens to become 0.0, meaning they are excluded from attention entirely.  

During this process, the token embedding vector gets projected into `Q`, `K`, and `V`, three linear projections. `Q`, or query, asks what the token is looking for. `K`, or key, is what each token has to offer. `V`, or value, is the actual content. The higher the relevancy (higher self-attention), the more of the value is being passed through.  

Once the attention layer is completed, we move on to a multilayer perceptron, or MLP. An MLP consists of 2 hidden layers that perform nonlinear analysis using neurons with trained weights. In the case of *nanoGPT*, it uses `GELU()`, or Gaussian Error Linear Unit.  
NOTE: MLP is a specific type of a feed-forward network, a network where data only flows in one direction: input --> hidden layers --> output.  

Combining attention and MLP together gives us one block of a transformer. Inside a block also exists two smaller components that ensure the model is as accurate as possible.  
Firstly, we have residual connections, which add the input back into the output. During backpropagation, gradients can become so small they "vanish". By returning `x + attention(x)` and `x + mlp(x)`, the original input is added back to the output of each sublayer, preventing gradients from disappearing or having zero influence.  
Secondly, we have LayerNorm, or layer normalization. It normalizes activations (values flowing through the network at any given point --> output of each layer). They are rescaled to have a mean of 0 and a standard deviation of 1. This is applied before the attention and MLP layers (known as pre-norm) to keep the values in a consistent range (therefore, these values cannot "explode" or "vanish").

**(5) Training Loop**

*nanoGPT* uses training and validation data to train and evaluate the model.
There are two modes: training and evaluation. By turning on training mode, the model enables dropout, which is a probabilistic function that zeroes out random activations during training to protect itself from overfitting to the data. Evaluation turns off dropout, disables gradient computation (`torch.no_grad()`), and calculates a loss function.  
In the case of *nanoGPT*, the loss function is called cross-entropy loss, which quantifies the difference between two probability distributions: the true value of the next token and the model's predictions. Therefore, the loss function that we use to evaluate the model's performance is the average of all the cross-entropy loss across all tokens in a batch.  

A forward pass orchestrates the full path of the data from input to output. In the forward pass, we create our input representation, which is `x = tok_emb + pos_emb`. Then, it runs `x` through the transformer block (attention, MLP, residual, LayerNorm) `n_layer` times. Once that is completed, `x = self.ln_f(x)`, applying a final LayerNorm to normalize the activations.  
Lastly, we have `logits = self.lm_head(x)`, a linear layer that projects the final hidden layer back up to `vocab_size`, producing a score (known as logit) for every token in the vocabulary. Logits represent the model's raw predictions for the next token.  

After the forward pass, the model calculates the loss function (see previous section for more details).  

Next, the model performs `loss.backward()`, which performs backpropagation and computes the gradients. They are each stored in the parameter's `.grad` attribute.  
Then, `step(optimizer)` reads the gradients and updates them to the weights, in effect saving them for the next forward pass.  
NOTE: an optimizer is an algorithm that decides how to update the weights given the gradients. *nanoGPT* uses `AdamW`, which adapts the learning rate and applies weight decay.

**(6) Sampling / Inference**

Sampling is defined as generating text from the model.  
*nanoGPT* generates text based on what is the most probable token in the sequence. The model receives a vector of how likely each token is with a probability float.
The probability float is the logit vector applied with softmax (meaning all logits now sum to 1.0).  

*nanoGPT* uses sampling hyperparameters to control the trade-off between diversity and coherence. For instance, always choosing the highest-value logit can create repetitive text. This is called greedy sampling. *nanoGPT* uses two hyperparameters: temperature and top-k.  
- Temperature: divides the logits by the temperature value (`logit / temperature value`) before softmax is applied. A low temperature makes the distribution sharper (model picks more confidently), while a high temperature offers more random outputs.
- Top-k: before sampling, it keeps only the top `k` most probable tokens and sets the rest to negative infinity (softmax treats negative infinity as zero). This prevents the model from sampling very unlikely tokens.

**(7) Key Design Decisions**

First: Karpathy chooses to set `wte.weight = lm_head.weight`. This is because the embedding table, `wte`, maps each token ID to a specific vector. In `lm_head` each vector maps back to a token ID. By setting those two to be equal, they would be sharing the same weight matrix. Tokens that are similar should have similar embeddings. Sharing the weights would enforce consistency between the input and output representations.

Secondly, we use pre-norm (applying LayerNorm before each sublayer) over post-norm (applying LayerNorm after each sublayer) because it produces more stable gradients during training.

Thirdly, by default, PyTorch uses 32-bit floats (`fp32`). In *nanoGPT*, Karpathy uses 16-bit floats (`fp16`) using `torch.autocast`. This halves memory usage and speeds up computation. However, gradients in `fp16` can become so small they underflow to zero. To fix this, we use `GradScaler`, which multiplies the loss by a large scale factor before backpropagation. It keeps the gradient in a representable fashion.  
NOTE: the gradient has to be divided back before the optimizer step.

Fourthly, instead of updating the weights after every single batch, the model accumulates the gradients over `grad_accum_steps` batches before updating it. This simulates a larger, effective batch size without needing more GPU memory.

Fifthly, Karpathy uses `AdamW` over `SGD` because `AdamW` is an adaptive optimizer. It maintains a separate learning rate for each parameter, adjusting it based on the history of past gradients, while `SGD` uses the same learning rate for all parameters. Additionally, `AdamW` applies weight decay to the weights directly, which helps regularization. For transformers, `AdamW` converges faster and more reliably than `SGD`.  
NOTE: weight decay is a regularization technique that penalizes large weights by shrinking them slightly after each update. This prevents the model from overfitting by keeping weights small.

Sixthly, `lm_head` is defined as `nn.Linear(n_embd, vocab_size, bias=False)`, where the bias is disabled. Bias adds a learned offset to each output. However, since it is fed into softmax right after, the constant added gets cancelled out and will have no effect. By turning bias off, it saves on parameters.

Lastly, Karpathy made sure to keep *nanoGPT* around ~300 lines per file. The idea is that anyone should be able to read the code and modify it without needing to untangle many layers of abstraction.

. . .

***Completely written by Justin Mo***