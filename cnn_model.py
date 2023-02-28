import numpy as nn
import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Defines CNN class. All input parameters should have
    consistent sizes or an error will be thrown. 
    Inputs:
    
    - linear_size: (flattened) size of the last MaxPool/dropout 
                   layer output, used to create single final
                   linear layer of size (linear_size, 5)
    - *layers: variable length. Each layer should be a list of 
               9 numbers. Each "layer" is made up of Conv2d ->
               ReLU -> MaxPool2d -> Dropout. The 9 numbers are
               1. Conv2d input channels
               2. Conv2d output channels
               3. Conv2d kernel_size
               4. Conv2d stride
               5. Conv2d padding
               6. MaxPool2d kernel_size
               7. MaxPool2d stride
               8. MaxPool2d padding
               9. Dropout parameter    
    """
    def __init__(self, linear_size, *layers):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential()
        name = 0
        for l in layers:
            self.cnn.add_module(str(name), nn.Conv2d(l[0], l[1], 
                                                     kernel_size = l[2], 
                                                     stride = l[3], 
                                                     padding = l[4])),
            name += 1
            self.cnn.add_module(str(name), nn.ReLU())
            name += 1
            self.cnn.add_module(str(name), nn.MaxPool2d(kernel_size = l[5], 
                                                        stride = l[6], 
                                                        padding = l[7]))
            name += 1
            self.cnn.add_module(str(name), nn.Dropout(l[8]))
            name += 1

        self.linear = nn.Linear(linear_size, 5) #5 labels in fmnist1
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1) #flatten to (N, 288)
        x = self.linear(x)
        return x
