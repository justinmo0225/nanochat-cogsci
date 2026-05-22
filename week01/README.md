# MNIST Digit Classifier

Neural network built with PyTorch that recognizes handwritten digits (0–9) using the MNIST dataset

***Justin's notes:*** this model is largely inspired by Andrej Karpathy's *micrograd* (tiny neural network engine) and the PyTorch *Quickstart* tutorial. After nearly 1,000 epochs, the accuracy is at 98.0% with an average loss of 0.065607  
Remember that any further pre-training may result in overfitting :p

## How It Works

1. **Data Loading** — Downloads 60,000 training images and 10,000 test images from MNIST
    - We parse them into tensors and normalize their values into a range from 0 to 1; this allows us to apply ReLU

2. **Model Architecture** — A fully connected neural network with 4 layers (2 hidden):
   - Input layer: 784 neurons (28x28 flattened image)
   - Hidden layer: 512 neurons + ReLU activation
   - Hidden layer: 512 neurons + ReLU activation
   - Output layer: 10 neurons (one per digit)

   This comes out to a total of (784 x 512) + (512 x 512) + (512 x 10) = 668,672 weights!  
   Also, there are 512 + 512 + 10 = 1,034 biases

   Total number of parameters used in this model is 668,672 + 1,034 = 669,706 parameters!

3. **Training** — We use Cross Entropy Loss and an SGD optimizer
    Forward pass -> loss function -> backpropogation -> update the weights

4. **Testing** — After each epoch, evaluates the model on unseen test data and reports accuracy and average loss

5. **Saving** — Model weights are saved to `model.pth` after training so they can be reloaded for future runs without retraining

## Training

Adjust the number of epochs in the script:
```python
epochs = 100
```
Then run:
```bash
python main.py
```

## Model Persistence
On first run, delete or skip the `model.load_state_dict(...)` line since no `model.pth` exists yet. After the first run, the weights are saved and future runs will continue training from where it left off.

## Requirements
The libraries you need to run this model are:
```bash
pip install torch torchvision
```