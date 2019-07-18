#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:16:57 2019

@author: server
"""

###################
### DATALOADING ###
###################

### Imports
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

### Defining data preprocessing step

# training set
train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

# test and valid sets
check_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

### Loading files as a dataset
trainset = datasets.ImageFolder('../data/train/', transform=train_transforms)
testset = datasets.ImageFolder('../data/test/', transform=check_transforms)
validset = datasets.ImageFolder('../data/valid/', transform=check_transforms)

### Creating batches of iterables from the datasets
trainloader = DataLoader(trainset, batch_size=24, shuffle=True)
testloader = DataLoader(testset, batch_size=24, shuffle=True)
validloader = DataLoader(validset, batch_size=24, shuffle=True)

loaders = {'train':trainloader, 'test':testloader, 'valid':validloader}

### Sanity check
# print(len(trainset), len(testset), len(validset))
# print(len(trainloader), len(testloader), len(validloader))
