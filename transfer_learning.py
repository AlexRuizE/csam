import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# Data
data_dir = '~/Downloads/'
cifar = torchvision.datasets.cifar.CIFAR100(
    root=data_dir,
    train=True,
    transform=None,
    target_transform=None,
    download=True)



# Load model



# Training params
num_classes = 2
batch_size = 8
num_epochs = 15
feature_extract = True



