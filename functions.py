#Useful functions for the method
import numpy as np
import torch
import torch.nn as nn
import torchvision

def get_normalized_fashion_mnist(val_size = 10000):
    """
    Get the Fashion MNIST dataset using torch.dataset and NORMALIZES
    
    Input:   val_size - how big the val set (default 10k), taken from training set, 
                        should be. Training set is 60k. 
                       
    Returns: trainxs - training data (28x28 images)
             trainys - training labels
             valxs   - validation data (28x28 images)
             valys   - validation labels
             testxs  - testing data (28x28 images)
             testys  - testing labels
    """
    index = 60000-val_size
    try:
        assert index > 30000
    except AssertionError as e:
        print('Validation set size must be less than 30000.')
    
    trainval = torchvision.datasets.FashionMNIST(root = "./data", 
                                             train = True,
                                             download = True, 
                                             transform = torchvision.transforms.ToTensor())
    test_set = torchvision.datasets.FashionMNIST(root = "./data", 
                                             train = False, 
                                             download = True, 
                                             transform = torchvision.transforms.ToTensor())

    #Create train/val/test xs and ys
    trainxs = trainval.data[:index] / 255.0 #normalize
    trainys = trainval.targets[:index]
    #Validation set is the last 10k of the training set
    valxs = trainval.data[index:] / 255.0 #normalize
    valys = trainval.targets[index:]
    testxs = test_set.data / 255.0 #normalize
    testys = test_set.targets
    
    return trainxs, trainys, valxs, valys, testxs, testys


def fmnist1(xs, ys):
    """
    Extracts dataset from MNIST data
    
    Input: data, labels
    
    Returns: torch dataset with all data points with labels
             0, 1, 4, 5 and 8 only. Also relabels to 0-4.
    """
    indices = np.where(np.in1d(ys, (0, 1, 4, 5, 8)))[0]
    out_ys = ys[indices]
    #Change labels to 0-4
    lab_dict = {0:0,1:1,4:2,5:3,8:4}
    for index in range(len(out_ys)):
        out_ys[index] = lab_dict[out_ys[index].item()]
    
    return torch.utils.data.TensorDataset(torch.unsqueeze(xs[indices], 1), out_ys)

def fmnist2(xs, ys):
    """
    Extracts dataset from MNIST data
    
    Input: data, labels
    
    Returns: torch dataset with all data points with labels
             2, 3, 6, 7 and 9 only. Also relabels to 0-4.
    """
    indices = np.where(np.in1d(ys, (2, 3, 6, 7, 9)))[0]
    out_ys = ys[indices]
    #Change labels to 0-4
    lab_dict = {2:0,3:1,6:2,7:3,9:4} 
    for index in range(len(out_ys)):
        out_ys[index] = lab_dict[out_ys[index].item()]
        
    return torch.utils.data.TensorDataset(torch.unsqueeze(xs[indices], 1), out_ys)

def get_iterators(train_dat, val_dat, test_dat, batch_s):
    """
    Gets iterators (torch Dataloader) for specific batch size
    
    Inputs:  train_dat - training data
             val_dat   - validation data
             test_dat  - test data
             batch_s   - batch size
            
    Returns: train_l   - training dataloader
             val_l     - validation dataloader
             test_l    - testing dataloader
    """
    #Define iterators of specified batch size
    train_l = torch.utils.data.DataLoader(train_dat, batch_size = batch_s, shuffle = True)
    val_l = torch.utils.data.DataLoader(val_dat, batch_size = batch_s, shuffle = False)
    test_l = torch.utils.data.DataLoader(test_dat, batch_size = batch_s, shuffle = False)
    return train_l, val_l, test_l
    
def get_accuracy(pred_y, label_y):
    """
  
    Inputs:  pred_y   - predicted labels
             labels_y - true labels
            
    Returns: accuracy
    
    """   
    
    pr = pred_y.argmax(dim = 1) #predicted label
    correct = torch.sum((pr == label_y).int()).item()
    return correct / len(label_y)
