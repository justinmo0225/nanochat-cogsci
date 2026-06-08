import torch
from gpt2 import GPT, device
from dataset import train_loader, val_loader

# hyperparameters
epochs = 3              # iterations through the dataset
grad_accum_steps = 4    # simulates 4 forward passes before gradient update; allows for larger effective batch size without needing more GPU memory
eval_interval = 500     # run validation every 500 steps
save_interval = 1000    # save weights every 1000 steps
learning_rate = 3e-4    # how fast the model updates its weights

model = GPT().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# fp16 uses 16 bits instead of 32 bits to represent numbers; reduces memory usage and speeds up training, but numbers may become too small to represent
# scaler — needed for fp16 training; prevents gradients from becoming too small to represent in half precision
scaler = torch.amp.GradScaler(enabled = device == 'cuda')

print(f"Training on {device}")

model.train()   # sets model to training mode, enabling dropout

for epoch in range(epochs):
    for step, (x, y) in enumerate(train_loader):    # grabs one batch of input (x) and target (y) sequences
        x = x.to(device)   # moves input batch to GPU
        y = y.to(device)   # moves target batch to GPU

        with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = device == 'cuda'):   # casts from fp32 to fp16 for faster training
            logits, loss = model(x, y)  # forward pass
            loss = loss / grad_accum_steps  # normalize loss since we're accumulating gradients

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:  # every 4 steps (add 1 due to zero-indexing)
            scaler.step(optimizer)  # updates all weights using accumulated gradients
            scaler.update()         # adjusts the scale factor so that small gradients do not underflow (go to zero) in fp16
            optimizer.zero_grad()   # clears gradient accumuluation since this batch is done
            print(f"epoch {epoch} step {step} | train loss: {loss.item():.4f}")  # prints loss value

        if (step + 1) % eval_interval == 0: # evaluate validation step every eval_interval steps
            model.eval()    # turns off dropout
            val_loss = 0
            
            with torch.no_grad():   # no need to compute gradients during evaluation since we evaluating it, not updating the weights
                for val_x, val_y in val_loader:
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                    with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = device == 'cuda'):   # fp16
                        _, loss = model(val_x, val_y)   # no need for logits as we are not generating/predicting
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)     # average loss across ALL validation batches
            print(f"epoch {epoch} step {step} | val loss: {val_loss:.4f}")
            model.train()   # turn dropout back on for training

        if (step + 1) % save_interval == 0: # saves weights every save_interval steps
            torch.save(model.state_dict(), f'checkpoint_epoch{epoch}_step{step}.pt')
            print(f"Saved checkpoint at epoch {epoch} step {step}")

torch.save(model.state_dict(), 'model_final.pt')
print("Training complete! Model saved as model_final.pt")