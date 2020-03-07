# Modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from src.func import *
import os
import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
# import time
# import copy



# Vars
model_name = 'alexnet'

# Model input size
input_size = initialize_model(model_name, init_model=False)

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = '/home/monorhesus/Data/oidv6/'
images = (i for i in os.listdir(f'{data_dir}/train/') if '.jpg' in i)



"""load image, returns cuda tensor"""
image = Image.open(data_dir+images[0])

plt.imshow(image)
plt.imshow(simple_loader(image).data.numpy().T)
plt.imshow(data_loader['train'](image).data.numpy().T)


image = data_loader['train'](image).float()

image = Variable(image, requires_grad=True)
image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
return image.cuda()  # assumes that you're using GPU

# Initialize model
initialize_model('resnet', num_classes, feature_extract, use_pretrained=True)


print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Model
m = models.squeezenet1_0(pretrained=True)



# Training params
num_classes = 2
batch_size = 8
num_epochs = 15
feature_extract = True



