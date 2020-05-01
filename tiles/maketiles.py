import torch
import os
import Augmentor
import matplotlib.pyplot as plt
from src.func import get_k_img, plot_rand_img
from torchvision import datasets, transforms
from torchvision.utils import make_grid


# Paths
oid_dir = '/home/monorhesus/Data/oidv6' # Path to Open Image Dataset root. Use toolkit.
out_dir = '/home/monorhesus/Pictures'

if out_dir[-1] == '/':
    out_dir=out_dir[:-1]

# Image set type with images (train or validation)
img_set = 'validation'

# Generate images with unequal paddings
img_dir = f'{oid_dir}/{img_set}'
classes = os.listdir(img_dir)

c = classes[0]
p = Augmentor.Pipeline(
    source_directory=f'{img_dir}/{c}',
    output_directory=f'{out_dir}/padded_imgs')
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(10)

# in_dir = '/home/monorhesus/Data/oidv6/' # Path to Open Image Dataset root. Use toolkit.
# out_dir = '/home/monorhesus/Pictures'
#
# img_transforms = transforms.Compose([
#     transforms.Pad(padding=(1000,100), fill=1,padding_mode='constant'),
#     transforms.Resize((512,512)),
#     transforms.ToTensor()])
# img_dataset = datasets.ImageFolder(torch.os.path.join(in_dir, img_dir),
#                                      transform=img_transforms)
#

plot_rand_img(img_dataset)
i = get_k_img(1, img_dataset)
plt.imshow(i[0].permute(1, 2, 0))
i[0]





# Make grid
img_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(p=.1),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((512,512)),
    transforms.ToTensor()])
img_dataset = datasets.ImageFolder(torch.os.path.join(oid_dir, img_dir),
                                   transform=img_transforms)

paddings = [0,100]
pad_values = [0,1]
for p in paddings:
    for v in pad_values:
        imgs = get_k_img(25, img_dataset)
        g = make_grid(imgs,
                      nrow=5,
                      padding=p,           # Size in pixels of padding
                      pad_value=v,          # 0 black, 1 white
                      normalize=False
                      )
        plt.imshow(g.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'{out_dir}/tiles_{p}pad_{v}bg.png', bbox_inches='tight', pad_inches=0)
        plt.close()
