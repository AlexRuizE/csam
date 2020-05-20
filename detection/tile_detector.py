"""Script to load convolutional neural network parameters"""

import torch
import torch.nn as nn
import argparse


# Args parse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', dest='param_dir', type=str, required=True, help='Path to model parameters.')
parser.add_argument('-o', '--outdir', dest='out_dir', type=str, required=True, help='Output directory for detected tile image paths.')
# parser.add_argument('-c', '--classes', dest='num_classes', type=int, required=False, default='2', help='Number of classes.')
# parser.add_argument('-m', '--model', dest='model_name', type=str, required=False, default='squeezenet', help='Model name.')
# parser.add_argument('-b', '--batch', dest='batch_size', type=int, required=False, default='16', help='Batch size.')
# parser.add_argument('-e', '--epochs', dest='num_epochs', type=int, required=False, default='10', help='Number of epochs.')
args = parser.parse_args()

# param_dir = args.param_dir
# out_dir = args.out_dir


param_dir = '/home/monorhesus/git/csam/'


tile_detector.pkl



