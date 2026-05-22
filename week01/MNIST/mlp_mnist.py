import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# forward pass --> loss function --> backpropogation --> optim (update weights)

# grabs the 60,000 training images from MNIST
training_data = datasets.MNIST(
    root = "data", train = True, download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)]),
    # converts data to tensor and scales it to the range [0, 1]
    # when scale = True, the pixel values are divided by 255 to normalize them to the range [0, 1].
)

# grabs the 10,000 test images from MNIST
test_data = datasets.MNIST(
    root = "data", train = False, download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)]),
)

batch_size = 64 # miniscule version of batch gradient descent

# DataLoader acts as an wrapper that takes raw data and feeds the data into the model in batches
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

# X is the image and y is the label (the digit that the image represents)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")    # [batch size, channels, height, width]
    print(f"Shape of y: {y.shape} {y.dtype}")       # should be a tensor of shape [batch size] and dtype int64
    break
    # ensures that the data is loaded properly

# always try to use GPU if available, otherwise fall back to CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# define the neural network!
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # calls parent nn.Module constructor
        self.flatten = nn.Flatten() # 28x28 image --> 784 length vector
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),    # first layer: input has 784 neurons and the output has 512 neurons
            nn.ReLU(),
            nn.Linear(512, 512),        # second layer: input has 512 neurons and the output has 512 neurons
            nn.ReLU(),
            nn.Linear(512, 10)          # last layer: input has 512 neurons and the output has 10 neurons

            # ReLU is our activation function: if a value is negative, set it to 0; otherwise, keep it as is
        )

    # forward pass
    def forward(self, x):
        x = self.flatten(x) # [64, 1, 28, 28] --> [64, 784]
        logits = self.linear_relu_stack(x)
        return logits   # raw, unnormalized scores for each class (the 10 digits)
        # always sum to 1.0; highest logit corresponds to the predicted class

model = NeuralNetwork().to(device)  # moves the entire model to the device so that it is faster
print(model)

# loads the weights from "model.pth" to save for future iterations
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only = True))

loss_fn = nn.CrossEntropyLoss() # measures how well how the prediction distribution matches the true label
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 60,000 images
    model.train()
    for batch, (X, y) in enumerate(dataloader): # counts from 0 to 937 (60000/64), which is the # of batches
        X, y = X.to(device), y.to(device)

        # computes prediction error
        pred = model(X)     # forward pass
        loss = loss_fn(pred, y)

        # backpropogate
        loss.backward()
        optimizer.step()    # takes a tiny step each time to adjust weights to reduce the loss
        optimizer.zero_grad()   # zeroes out the gradients to prevent accumulation

        # gives us loss value every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d} / {size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)      # 10,000 images total
    num_batches = len(dataloader)       # 10,000 / 64 = 156 batches
    model.eval()    # OPPOSITE of model.train()
    test_loss = 0   # accumulates total loss across all batches
    correct = 0
    with torch.no_grad():   # no need to update weights --> no need for gradients
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) # better and updated forward pass
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches    # gets average loss/batch
    correct /= size     # average accuracy
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# ITERATES THROUGH THE TRAINING AND TESTING PROCESS FOR [epoch] TIMES
# Update [epoch] to suit your training preferences
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

epochs = 100

# actively trains the model [epoch] times
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model.eval()    # built-in function; sets model to evaluation mode (turns off dropout and batch normalization)

# saves the model's weights
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")