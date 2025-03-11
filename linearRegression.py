import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

def train():
    torch.manual_seed(42)

    # Set the number of epochs (how many times the model will pass over the training data)
    epochs = 100
    
    # Create empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    
    for epoch in range(epochs):
        ### Training
    
        # Put model in training mode (this is the default state of a model)
        model_0.train()
    
        # 1. Forward pass on train data using the forward() method inside 
        y_pred = model_0(X_train)
        # print(y_pred)
    
        # 2. Calculate the loss (how different are our models predictions to the ground truth)
        loss = loss_fn(y_pred, y_train)
    
        # 3. Zero grad of the optimizer
        optimizer.zero_grad()
    
        # 4. Loss backwards
        loss.backward()
    
        # 5. Progress the optimizer
        optimizer.step()
    
        ### Testing
    
        # Put the model in evaluation mode
        model_0.eval()
    
        with torch.inference_mode():
          # 1. Forward pass on test data
          test_pred = model_0(X_test)
    
          # 2. Caculate loss on test data
          test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
    
          # Print out what's happening
          if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
