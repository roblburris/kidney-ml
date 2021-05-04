import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, init_input):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(init_input, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 3),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)