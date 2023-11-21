import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNN(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3)
        
        self.fc1 = nn.Linear(576, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = self.conv1(x) # 28*28*8
        x = F.relu(F.max_pool2d(x, 2)) # 14*14*8
        x = self.conv2(x) # 12*12*16
        x = F.relu(F.max_pool2d(x, 2)) # 6*6*16
        
        x = x.view(-1, 576)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        x = F.softmax(x, dim=1)
        
        return x


def accuracy(outputs:np.ndarray, targets:np.ndarray) -> float:
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: the probability of the labels predicted by the net
        targets: the actual target labels

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==targets) / float(targets.size)


def recall(outputs:np.ndarray, targets:np.ndarray) -> float:
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: the labels predicted by the net
        targets: the actual targets 

    Returns: (float) accuracy in [0,1]
    """
    pass


loss_fn = F.cross_entropy

metrics = {'accuracy': accuracy}
        
