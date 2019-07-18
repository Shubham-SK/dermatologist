#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:15:54 2019

@author: server
"""

##########################
### MODEL ARCHITECTURE ###
##########################

### Imports
from torchvision import models
import torch.nn as nn

### Importing the model (inception v3 CNN) pretrained on Imagenet
Classifier_net = models.inception_v3(aux_logits=False)

### Modifying the fully connected layers to fit our dataset instead of imagenet
# Note: Modify final fully connected network by using nn.Sequential() to define
#       layers

Classifier_net.fc = nn.Linear(2048, 3)
