# ml_model.py
import torch
import torch.nn as nn

class SnakeNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=4):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x