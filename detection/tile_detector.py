"""Script to load convolutional neural network parameters"""

import torch
import torch.nn as nn
import argparse
import os
from torchvision import models, transforms, datasets
from sklearn.metrics import confusion_matrix
from shutil import copyfile

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
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    else:
        raise RuntimeError('Model not available.')

    return model_ft, input_size



# Args parse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mdir', dest='param_dir', type=str, required=True, help='Path to model parameters.')
parser.add_argument('-i', '--idir', dest='image_dir', type=str, required=True, help='Path to images for classification.')
parser.add_argument('-o', '--odir', dest='out_dir', type=str, required=True, help='Output directory for detected tile image paths.')
args = parser.parse_args()

# param_dir = args.param_dir
# image_dir = args.image_dir
# out_dir = args.out_dir

param_dir = '/home/monorhesus/Desktop/'
image_dir = '/home/monorhesus/Pictures/c3p_mock/'
out_dir = '/home/monorhesus/Pictures/'

# for testing
empty_mock = '/home/monorhesus/Pictures/empty_mock/'

# Rearrange image dir structure to fit ImageLoader (temporary hack)
print("Rearranging image dir structure, this might take a while...")

# image_dir_files = os.listdir(image_dir)
image_dir_files = os.listdir(empty_mock)

assert len(image_dir_files) != 0, "Image directory is empty."

if len(image_dir_files) > 1:

os.path.isdir(image_dir_files)


for i in class_examples:
    src = os.path.join(in_path, i)
    dst = os.path.join(out_path, i)
    copyfile(src, dst)



# Load pre-trained parameters
model_name = 'squeezenet'
num_classes = 2
feature_extract = True
model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
model.load_state_dict(torch.load(param_dir + 'tile_detector.pkl'))
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


# # Get tile indices
# img_names = [i[0] for i in img_dataset.imgs]
# tile_indices = torch._np.array([1 if 'tile_' in i else 0 for i in img_names])
#
#
#
# confusion_matrix(tile_indices, results)