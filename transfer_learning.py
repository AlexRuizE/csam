# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

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


#
posible_models = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
model_name = posible_models[3]


data_dir = '/home/monorhesus/Data/hymenoptera_data'

num_classes = 2

batch_size = 8

num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,True we only update the reshaped layer params
feature_extract = True



