#Function for training the model
import numpy as np
import torch
import torch.nn as nn
from cnn_model import CNN
from functions import get_accuracy

def train_epoch(model, opt, loss_fn, train_iter, device):
    """
    For training a model for an epoch.
    
    Inputs:  model      - a model class
             opt        - optimizer
             loss_fn    - loss function
             train_iter - training dataloader
             device     - e.g. GPU
            
    Returns: loss, accuracy
    
    """
    total_loss = 0.0
    total_acc = 0.0
    model.train()
    
    for imgs, labels in train_iter:
        imgs, labels = imgs.to(device), labels.to(device)
    
        opt.zero_grad()

        train_output = model(imgs.float())
        loss = loss_fn(train_output, labels)
        acc = get_accuracy(train_output, labels)

        loss.backward()
        opt.step()
        total_loss += loss.item()
        total_acc += acc
        
    return total_loss / len(train_iter), total_acc / len(train_iter) 

def evaluation(model, loss_fn, eval_iter, device):
    """
    Evaluates model.
    
    Inputs:  model     - model to be evaluated
             loss_fn   - loss function
             eval_iter - dataloader of set to be evaluated
             device    - e.g. GPU
             
    Returns: loss, accuracy
    
    """
    #Evaluate model for iterator of set to be evaluated
    #Return loss, accuracy
    total_loss = 0.0
    total_acc = 0.0
    model.eval()
    
    batch_vout = list()
    
    with torch.no_grad():
        for imgs, labels in eval_iter:
            imgs, labels = imgs.to(device), labels.to(device)
            
            eval_output = model(imgs.float())
            loss = loss_fn(eval_output, labels)
            acc = get_accuracy(eval_output, labels)
            total_loss += loss.item()
            total_acc += acc
            
    return total_loss / len(eval_iter), total_acc / len(eval_iter)

def train_cnn_model(train_ldr, val_ldr, l_rate, linear_size, device, loss_fn,
                    *layers):
    """
    For training a CNN model.
    
    Inputs:   train_ldr   - training dataloader
              val_ldr     - validation dataloader
              l_rate      - learning rate
              linear_size - size of the linear layer in the CNN model
              device      - e.g. GPU
              loss_fn     - e.g. nn.CrossEntropyLoss()
              layers      - specified layers of the CNN
            
    Returns:  train_losses, train_accuracies, val_losses, val_accuracies, 
              model
              
    """
    
    model = CNN(linear_size, *layers)
    model.to(device)
    
    #optimizer - specifies that only updates gradients which require grad (e.g. embedding)
    opt = torch.optim.SGD(model.parameters(), lr = l_rate)
    
    train_losses = list()
    train_accuracies = list()
    val_losses = list()
    val_accuracies = list()
    
    #Initial training loss
    old_loss, _ = evaluation(model, loss_fn, train_ldr, device)
    
    #Run until training loss converges
    count = 0
    CONV = 1e-5 #convergence threshold
    while True:
        count += 1
              
        train_loss, train_acc = train_epoch(model, opt, loss_fn, train_ldr, device)
        val_loss, val_acc = evaluation(model, loss_fn, val_ldr, device)
    
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch: {count}, training loss: {train_loss}, validation loss: {val_loss}')
        
        #Establish exit condition based on conv threshold
        if(abs(old_loss - train_loss) < CONV):
            break
        old_loss = train_loss
    
    return train_losses, train_accuracies, val_losses, val_accuracies, model
