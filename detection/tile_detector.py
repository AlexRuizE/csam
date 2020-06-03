"""Script to load convolutional neural network parameters"""

import torch
import torch.nn as nn
import argparse
import os
from torchvision import models, transforms, datasets
from sklearn.metrics import confusion_matrix
from shutil import move

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    if model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    else:
        raise RuntimeError('Model not available.')
    return model_ft, input_size

def dir_structure(image_dir):
    """Creates the appropriate mock image dir struture, if needed."""
    image_dir_files = os.listdir(image_dir)
    assert len(image_dir_files) != 0, "Image directory is empty."
    if len(image_dir_files) == 1:
        f = ''.join([image_dir, os.sep, image_dir_files[0]])
        if os.path.isdir(f):
            print('Image directory structure test: OK.')
    else:
        print("Rearranging image dir structure, this might take a while...")
        fake_dir = '/1'
        os.mkdir(image_dir+fake_dir)
        for i in image_dir_files:
            src = os.path.join(image_dir, i)
            dst = os.path.join(image_dir+fake_dir, i)
            move(src, dst)
        print("Done. Directory structure is now OK.")


# Args parse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mdir', dest='param_dir', type=str, required=True, help='Path to model parameters.')
parser.add_argument('-i', '--idir', dest='image_dir', type=str, required=True, help='Path to images for classification.')
parser.add_argument('-o', '--odir', dest='out_dir', type=str, required=True, help='Output directory for detected tile image paths.')
parser.add_argument('-f', '--mfile', dest='model_file', type=str, required=False, help='Custom model name (params) file.')
args = parser.parse_args()

# param_dir = args.param_dir
# image_dir = args.image_dir
# out_dir = args.out_dir
# model_file = args.model_file

param_dir = '/home/monorhesus/Desktop/'
image_dir = '/home/monorhesus/Pictures/c3p_mock/'
out_dir = '/home/monorhesus/Pictures/'
model_file = '/tile_detector_1bis.pkl'

# Clean-up paths
if param_dir[-1] == '/':
    param_dir=param_dir[:-1]
if image_dir[-1] == '/':
    image_dir=image_dir[:-1]
if out_dir[-1] == '/':
    out_dir=out_dir[:-1]
if model_file[0] == '/':
    model_file = model_file[1:]

# for testing
# image_dir = '/home/monorhesus/Pictures/empty_mock/'
# image_dir = '/home/monorhesus/Pictures/csam_test_tiles'
image_dir = '/home/monorhesus/Pictures/c3p_mock/'

# Rearrange image dir structure to fit ImageLoader (temporary hack)

# image_dir_files = os.listdir(image_dir)
# image_dir_files = os.listdir(empty_mock)


dir_structure(image_dir)

# Load pre-trained parameters
model_name = 'squeezenet'
num_classes = 2
feature_extract = True
model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
model.load_state_dict(torch.load(os.path.join(param_dir, model_file)))
model.eval()


# Data augmentation and normalization for training
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# TODO: datasets.FlatImageFolder
img_dataset = datasets.ImageFolder(
                image_dir,
                transform=data_transforms)

batch_size = 16
dataloader = torch.utils.data.DataLoader(img_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4)

softmax = torch.nn.Softmax(dim=1)
results = torch._np.array([])

for inputs, labels in dataloader:
    outputs = model(inputs) # logits
    class_probs = softmax(outputs)
    pred_class = torch.max(class_probs,1)[1].numpy()
    results = torch._np.append(results, pred_class)
    # print(outputs)


# Get tile indices
img_names = [i[0] for i in img_dataset.imgs]
tile_indices = torch._np.array([1 if 'tile_' in i else 0 for i in img_names])

print(confusion_matrix(tile_indices, results))