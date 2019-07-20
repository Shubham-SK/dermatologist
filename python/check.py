#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:07:08 2019

@author: server
"""

####################################
### Training and Testing Methods ###
####################################

import numpy as np
import torch

### Training method
def train(epochs, loaders, model, optimizer, criterion, device, save_path, verbose=True):
    """
    Name: train

    Parameters: 8
    _____________

    epochs: int, # of training iterations
    loaders: dictionary, consists of trainloader, testloader and validloader
    model: lmao the model
    optimizer: torch.optim.{optimizer}
    criterion: torch.nn.{loss}
    device: str, hardware accelarator
    save_path: str, location to save best model in
    verbose: bool, decides whether to print out model progress, default True
    """
    ### Moving model to hardware accelerator
    model.to(device)

    valid_loss_min = 0.8294758399327596 #Lowest Achieved.

    ### Training and Validation Loop
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

	# Train loop
        model.train()
        for batch_idx, (feature, label) in enumerate(loaders['train']):
            feature, label = feature.to(device), label.to(device)

            optimizer.zero_grad()

            log_ps, aux_outputs = model(feature)
            loss1 = criterion(log_ps, label)
            loss2 = criterion(aux_outputs, label)
            loss = loss1 + 0.4 * loss2

            loss.backward()
            optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (loss.item() - train_loss)

        # Validation loop
        model.eval()
        for batch_idx, (feature, label) in enumerate(loaders['valid']):
            feature, label = feature.to(device), label.to(device)

            log_ps = model(feature)
            loss = criterion(log_ps, label)

            valid_loss += (1 / (batch_idx + 1)) * (loss.item() - valid_loss)

        # Print out results
        if verbose:
            print('Epoch: {} \tTraining Loss: {} \tValidation Loss: {}'.format(e + 1, train_loss, valid_loss))
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            print('Saving Model...')

    return model

### Testing method
def test(loaders, model, criterion, device, verbose=True):
    """
    Name: test

    Parameters: 5
    _____________

    loaders: dictionary, consists of trainloader, testloader and validloader
    model: lmao the model
    criterion: torch.nn.{loss}
    device: str, hardware accelarator
    verbose: bool, decides whether to print out model progress, default True
    """
    ### Moving model to hardware accelerator
    model.to(device)

    # Initializing test results
    test_loss = 0.0
    correct = 0
    total = 0

    # Test loop
    model.eval()
    for batch_idx, (feature, label) in enumerate(loaders['test']):
        feature, label = feature.to(device), label.to(device)

        log_ps = model(feature)
        loss = criterion(log_ps, label)

        test_loss += (1 / (batch_idx + 1)) * (loss.item() - test_loss)

        pred = log_ps.data.max(1, keepdim=True)[1]
        correct += np.sum((label.t()[0] == pred).cpu().numpy())
        total += label.shape[0]

    if verbose:
        print('Test Loss: {}\n'.format(test_loss))
        print('Test Accuracy: {}%, ({}/{})'.format(100 * correct / total, correct, total))

