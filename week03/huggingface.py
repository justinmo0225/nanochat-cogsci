from transformers import GPT2Model
hf = GPT2Model.from_pretrained('gpt2')
for name, param in hf.named_parameters():
    print(name, param.shape)